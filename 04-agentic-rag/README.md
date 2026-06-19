# Agentic RAG

The LLM **plans its own retrieval strategy** — deciding which tools to use, in what order, and when it has enough information to answer.

## The Problem with Fixed Pipelines

In every other RAG pattern, the retrieval logic is hardcoded. Every query goes through the same steps regardless of what the question actually needs:

| Query | What you need | What naive/hybrid RAG does |
|-------|--------------|--------------------------|
| "What is SpaceX?" | One lookup | Runs the full pipeline anyway |
| "Compare SpaceX and NASA for Mars" | Two retrievals + synthesis | Same pipeline, likely misses one |
| "ISS speed in mph?" | Retrieval + unit conversion | Can't do math |

An agent adapts. It reads the question, picks the right tools, and iterates.

## The Pattern — ReAct Loop

**ReAct = Reason + Act**

```
Question
   │
   ▼
┌─────────────────────────────────────────────┐
│  LLM reasons: "What do I need to answer?"  │
│  LLM picks:   tool_name + arguments         │
└────────────────────┬────────────────────────┘
                     │ tool call
                     ▼
              Tool executes
              (search / lookup / calculate)
                     │ result
                     ▼
┌─────────────────────────────────────────────┐
│  LLM reasons: "Is this enough?"             │
│  If yes  → final answer                     │
│  If no   → call another tool                │
└─────────────────────────────────────────────┘
```

### Tools in This Project

| Tool | What it does | When the agent uses it |
|------|-------------|----------------------|
| `search_docs` | Semantic search over document corpus | General questions, narratives |
| `get_entity` | Exact entity lookup (SpaceX, NASA, ISS...) | Precise facts about known entities |
| `calculate` | Safe arithmetic evaluator | Unit conversions, math |

## When to Use It

✅ Questions require **multiple retrieval steps** to answer  
✅ Queries are **ambiguous** — you don't know upfront what to retrieve  
✅ You need to **combine retrieval with computation or other actions**  
✅ Users ask follow-up or exploratory questions  

## How to Run

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## What to Look At in the Code

1. **`TOOLS`** — how tools are described to the LLM in OpenAI function-calling format
2. **`run_agent()`** — the ReAct loop: call LLM → execute tool → feed result back
3. **`calculate()`** — safe AST-based evaluator (no `eval()`)
4. The third query — watch the agent chain `get_entity` (to get km/s speed) + `calculate` (to convert to mph)

## Dependencies

- `chromadb` + `sentence-transformers` — for the `search_docs` tool
- `openai` — LLM client with function calling
