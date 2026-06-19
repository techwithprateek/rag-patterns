"""
Agentic RAG — the LLM plans its own retrieval strategy.

The Problem with Fixed Pipelines:
  In Naive/Hybrid/GraphRAG, the retrieval strategy is hardcoded. Every query
  goes through the same pipeline regardless of what the question actually needs.

  But questions vary:
    "What is SpaceX?" → single lookup, done
    "Compare SpaceX and NASA's approach to Mars" → needs two retrievals + synthesis
    "What is the orbital speed of the ISS in miles per hour?" → needs retrieval + math

  A static pipeline can't adapt. An agent can.

The Agentic Approach:
  We implement a ReAct loop (Reason + Act) where the LLM:
    1. Reads the question
    2. Decides which tool to use and with what input
    3. Sees the tool's output
    4. Decides whether it has enough to answer, or needs another tool call
    5. Repeats until it can give a final answer

Tools available to the agent:
  - search_docs(query)     → vector search over a document corpus
  - get_entity(name)       → exact lookup of a known entity (like a mini-graph)
  - calculate(expression)  → evaluate a math expression safely
"""

import os
import json
import ast
import operator
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────
#  LLM abstraction
# ─────────────────────────────────────────────────────────────────

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "groq":
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        ), "llama-3.1-8b-instant"
    if provider == "ollama":
        return OpenAI(api_key="ollama", base_url="http://localhost:11434/v1"), \
               os.getenv("OLLAMA_MODEL", "llama3.2")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────
#  Knowledge base — indexed for search_docs tool
# ─────────────────────────────────────────────────────────────────

DOCUMENTS = [
    "SpaceX was founded in 2002 by Elon Musk with the goal of making space travel cheaper "
    "and eventually colonizing Mars. Its Falcon 9 is the world's first orbital-class reusable rocket.",

    "NASA was founded in 1958 and is a US government agency. Its Mars missions include "
    "Curiosity (2012) and Perseverance (2021). NASA's SLS rocket is expendable, not reusable.",

    "The International Space Station (ISS) orbits Earth at 408 km altitude and travels "
    "at 7.66 km/s (27,576 km/h or 17,132 mph). It completes one orbit every 92 minutes.",

    "SpaceX's Starship is a fully reusable spacecraft designed for Mars missions, lunar landings, "
    "and point-to-point Earth travel. Its first successful orbital flight was in 2024.",

    "The Artemis program is NASA's plan to return humans to the Moon. Artemis 1 (2022) was "
    "uncrewed. Artemis 2 (2024) will be the first crewed test. Moon landing planned for Artemis 3.",

    "Mars is approximately 225 million km from Earth on average. A one-way trip with current "
    "propulsion takes 7-9 months. SpaceX aims to cut this with Starship's higher thrust-to-weight ratio.",

    "Blue Origin is a space company founded by Jeff Bezos in 2000. Its New Shepard rocket does "
    "suborbital tourism flights. Its New Glenn rocket is designed for orbital launches.",

    "Orbital velocity is the minimum speed needed to stay in orbit. At 400 km altitude, "
    "this is approximately 7.66 km/s. Below this, the spacecraft would fall back to Earth.",
]

# Build in-memory vector index for the search_docs tool
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("agent_kb")
collection.add(
    documents=DOCUMENTS,
    embeddings=embedding_model.encode(DOCUMENTS).tolist(),
    ids=[f"doc_{i}" for i in range(len(DOCUMENTS))]
)

# Entity lookup table for the get_entity tool
ENTITIES = {
    "spacex":    "SpaceX: Private space company by Elon Musk. Falcon 9 (reusable), Starship (Mars). Founded 2002.",
    "nasa":      "NASA: US government space agency. Artemis (Moon), Perseverance (Mars). Founded 1958.",
    "iss":       "ISS: Orbits at 408km, speed 7.66 km/s (17,132 mph), 92-min orbit. Inhabited since 2000.",
    "blue origin": "Blue Origin: Jeff Bezos' space company. New Shepard (suborbital), New Glenn (orbital). Founded 2000.",
    "starship":  "Starship: SpaceX's fully reusable Mars rocket. First orbital flight 2024. Largest rocket ever built.",
}

print("✅ Agent knowledge base ready.\n")


# ─────────────────────────────────────────────────────────────────
#  Tool definitions
# ─────────────────────────────────────────────────────────────────

# Tool specs in OpenAI function-calling format.
# The agent (LLM) reads these to know what tools exist and how to call them.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the knowledge base using semantic similarity. Use for general questions about space, rockets, missions, or orbit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity",
            "description": "Look up a specific named entity (SpaceX, NASA, ISS, Blue Origin, Starship). Use when you need precise facts about a known entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Entity name to look up"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Use for unit conversions or arithmetic. Example: '7.66 * 3600' to convert km/s to km/h.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A safe arithmetic expression using +, -, *, /, (, ), and numbers"}
                },
                "required": ["expression"]
            }
        }
    }
]


# ─────────────────────────────────────────────────────────────────
#  Tool implementations
# ─────────────────────────────────────────────────────────────────

def search_docs(query: str) -> str:
    """Semantic search over the knowledge base."""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=2)
    chunks = results["documents"][0]
    return "\n---\n".join(chunks)


def get_entity(name: str) -> str:
    """Exact entity lookup."""
    result = ENTITIES.get(name.lower().strip())
    return result if result else f"No entity found for '{name}'. Try: {list(ENTITIES.keys())}"


def calculate(expression: str) -> str:
    """
    Safe arithmetic evaluator — no eval(), no exec().
    Only allows numbers and basic operators to prevent code injection.
    """
    # Only allow digits, spaces, and basic math operators
    allowed_chars = set("0123456789 +-*/().")
    if not all(c in allowed_chars for c in expression):
        return "Error: Only basic arithmetic is allowed (+, -, *, /, parentheses, numbers)."
    try:
        # Parse to AST and evaluate node-by-node — never calls eval()
        tree = ast.parse(expression, mode="eval")
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.USub: operator.neg
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BinOp):
                return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
            if isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](eval_node(node.operand))
            raise ValueError(f"Unsupported operation: {type(node)}")

        result = eval_node(tree.body)
        return f"{expression} = {result:.4f}"
    except Exception as e:
        return f"Calculation error: {e}"


def dispatch_tool(name: str, args: dict) -> str:
    """Route a tool call from the agent to the right implementation."""
    if name == "search_docs":
        return search_docs(args["query"])
    if name == "get_entity":
        return get_entity(args["name"])
    if name == "calculate":
        return calculate(args["expression"])
    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────
#  The ReAct Agent Loop
# ─────────────────────────────────────────────────────────────────

def run_agent(question: str, max_iterations: int = 5) -> str:
    """
    Runs the Reason + Act (ReAct) loop.

    Each iteration:
      1. Send the conversation history to the LLM
      2. If the LLM calls a tool → execute it, append result, continue
      3. If the LLM gives a text response → that's the final answer, stop

    max_iterations prevents infinite loops if the agent gets stuck.
    """
    client, model = get_llm()

    # Conversation starts with a system prompt + the user's question
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant with tools to look up space industry facts. "
                "For each question, think about what information you need, use your tools to "
                "retrieve it, and synthesize a clear, factual answer. "
                "If a question requires multiple lookups or calculations, do them step by step."
            )
        },
        {"role": "user", "content": question}
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"  # let the LLM decide whether to call a tool
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # ── Case 1: LLM wants to call a tool ─────────────────────
        if finish_reason == "tool_calls" and message.tool_calls:
            messages.append(message)  # add assistant message with tool call

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"   🔧 Tool call [{iteration+1}]: {tool_name}({tool_args})")
                result = dispatch_tool(tool_name, tool_args)
                print(f"      Result: {result[:120]}...")

                # Append the tool result so the LLM sees it in the next iteration
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        # ── Case 2: LLM has a final answer ────────────────────────
        else:
            return message.content

    return "Agent reached maximum iterations without a final answer."


# ─────────────────────────────────────────────────────────────────
#  Run example queries
# ─────────────────────────────────────────────────────────────────

questions = [
    # Single-hop: one tool call should be enough
    "What is SpaceX's approach to rocket reusability?",

    # Multi-hop: needs two entity lookups then synthesis
    "Compare SpaceX and NASA's plans for Mars exploration.",

    # Tool-chaining: retrieve + calculate
    "What is the ISS orbital speed in miles per hour?",
]

for question in questions:
    print(f"❓ {question}")
    answer = run_agent(question)
    print(f"   💬 {answer}\n")
