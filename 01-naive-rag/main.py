"""
Naive RAG — the simplest form of Retrieval-Augmented Generation.

Pattern:
  1. Chunk documents into small pieces
  2. Embed each chunk into a vector
  3. Store vectors in a vector database
  4. At query time: embed the query, find the closest chunk vectors
  5. Pass those chunks as context to the LLM

This is the baseline. It's quick to build and works well for small,
clean corpora where semantic similarity reliably maps to relevance.
"""

import os
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────
#  LLM abstraction — same code works with OpenAI, Groq, or Ollama
# ─────────────────────────────────────────────────────────────────

def get_llm():
    """
    Returns (client, model_name) for the configured provider.
    All three providers expose an OpenAI-compatible API, so the
    rest of the code doesn't need to change per provider.
    """
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "groq":
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        return client, "llama-3.1-8b-instant"

    if provider == "ollama":
        # Ollama runs locally — no API key needed
        client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        return client, os.getenv("OLLAMA_MODEL", "llama3.2")

    # Default: OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client, "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────
#  Knowledge base — the documents we want to search over
# ─────────────────────────────────────────────────────────────────

# In a real system, these would come from PDFs, databases, or APIs.
# Here we use a small set of space exploration facts so you can run
# this without any external data source.
DOCUMENTS = [
    "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969. "
    "Neil Armstrong and Buzz Aldrin walked on the surface while Michael Collins orbited above.",

    "The James Webb Space Telescope (JWST) launched on December 25, 2021. "
    "It observes in infrared light, letting it see through dust clouds and capture images of the earliest galaxies.",

    "Mars has two small moons called Phobos and Deimos. "
    "Scientists believe they are captured asteroids, not moons that formed alongside the planet.",

    "Voyager 1 is the most distant human-made object ever launched. "
    "It entered interstellar space in 2012 and continues to transmit data back to Earth from over 23 billion km away.",

    "SpaceX's Falcon 9 is a partially reusable rocket. "
    "The first stage booster autonomously lands back on Earth or a drone ship and is refurbished for future flights.",

    "The International Space Station (ISS) orbits Earth at ~400 km altitude and travels at roughly 28,000 km/h. "
    "It has been continuously inhabited since November 2000.",

    "Saturn's rings are made mostly of water ice and rocky debris, ranging from microscopic grains to chunks the size of a house. "
    "Despite spanning hundreds of thousands of kilometers, the rings are only about 10 meters thick in some places.",

    "The Hubble Space Telescope has been operating since 1990. "
    "Its deep-field images revealed thousands of galaxies in a patch of sky that appeared completely empty to the naked eye.",
]


# ─────────────────────────────────────────────────────────────────
#  Step 1 — Embed and index all documents
# ─────────────────────────────────────────────────────────────────

# sentence-transformers runs locally — no embedding API key required.
# all-MiniLM-L6-v2 is small (~80MB) but good enough for most tasks.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB in-memory client — no disk, no server, no setup.
# For production, swap this for chromadb.PersistentClient() or a hosted DB.
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("space_facts")

print("📥 Indexing documents...")
doc_embeddings = embedding_model.encode(DOCUMENTS).tolist()

collection.add(
    documents=DOCUMENTS,
    embeddings=doc_embeddings,
    ids=[f"doc_{i}" for i in range(len(DOCUMENTS))]
)
print(f"✅ Indexed {len(DOCUMENTS)} documents.\n")


# ─────────────────────────────────────────────────────────────────
#  Step 2 — Retrieve relevant chunks for a query
# ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> list[str]:
    """
    Embeds the query and finds the top-k most semantically similar chunks.

    This is pure cosine similarity in vector space — there's no keyword
    matching, no filters, no reranking. Simple, but fragile at scale.
    """
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # results["documents"] is a list-of-lists (one per query),
    # so we take index [0] for our single query
    return results["documents"][0]


# ─────────────────────────────────────────────────────────────────
#  Step 3 — Generate an answer using the retrieved context
# ─────────────────────────────────────────────────────────────────

def generate(query: str, context_chunks: list[str]) -> str:
    """
    Passes the query + retrieved chunks to the LLM.

    The system prompt instructs the LLM to answer ONLY from the provided
    context. Without this constraint the model might blend retrieved facts
    with its own training data, which defeats the purpose of RAG.
    """
    client, model = get_llm()

    # Join chunks into a numbered list for readability
    context = "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(context_chunks))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual assistant. Answer the user's question using ONLY "
                "the context provided. If the answer is not in the context, say "
                "'I don't have that information.'"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0  # deterministic — we want factual answers, not creative ones
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────
#  Run the full RAG pipeline on a few example queries
# ─────────────────────────────────────────────────────────────────

queries = [
    "When did the first humans land on the Moon?",
    "How fast does the ISS travel?",
    "What are Saturn's rings made of?",
    "Who invented the telephone?",  # not in our knowledge base — watch what happens
]

for query in queries:
    print(f"❓ {query}")

    chunks = retrieve(query)
    print(f"   Retrieved chunks:")
    for chunk in chunks:
        # Print just the first 90 chars so the output stays readable
        print(f"     • {chunk[:90]}...")

    answer = generate(query, chunks)
    print(f"   💬 {answer}\n")
