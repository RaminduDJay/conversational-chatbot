# Project Overview: Conversational Podcast Chatbot

## What it does
This project is a Streamlit-based chatbot that answers questions about podcast
transcripts. It behaves like a podcast host, adds source citations with
timestamps, and can use retrieval-augmented generation (RAG) when the answer
is not present in prior conversation history.

## Main components
- UI + session management: `app.py` runs the Streamlit app, manages user
  sessions, and stores chat history in Zep.
- Entry classifier: `agents/entry.py` uses an LLM to decide whether the answer
  can be produced from history or whether RAG should be used. It outputs:
  `answer`, `use_rag`, `user_intent`, `output_emotion`.
- RAG pipeline: `agents/podcast_agent.py` orchestrates retrieval and answer
  generation with quality checks.
- Vector DB and search: `agents/rag_db.py` connects to Qdrant, creates
  embeddings, extracts query entities, and performs hybrid search.
- RAG helpers: `agents/rag_function.py` implements document retrieval,
  grading, answer generation, hallucination checks, and query reformulation.
- Long-term memory: `agents/memory_db.py` integrates with Zep for session
  memory and conversation summaries.
- API clients: `agents/clients.py` initializes OpenAI chat and embeddings clients.

## High-level flow
1. User sends a message in the Streamlit UI (`app.py`).
2. The entry LLM (`agents/entry.py`) decides whether to answer from history or
   use RAG.
3. If `use_rag=False`, the answer is returned directly.
4. If `use_rag=True`, `agents/podcast_agent.py`:
   - retrieves documents from Qdrant,
   - grades document relevance,
   - generates an answer in a podcast-host tone,
   - checks hallucinations and answer relevance,
   - retries with a reformulated query if needed.
5. The final answer is stored in Zep, and a summary is added to short-term
   history for future context.

## Environment variables
Set these in a `.env` file:
- `QDRANT_CLOUD_API_KEY`
- `OPENAI_API_KEY`
- `ZEP_API_KEY`
- Optional: `QDRANT_URL`, `EMBEDDING_MODEL`, `VECTOR_SIZE`, `ENTITY_MODEL`

## Running locally
```bash
pip install uv
uv run streamlit run app.py
```

## Data ingestion
The vector database is populated from podcast transcripts using the notebook:
`notebooks/data_injest.ipynb`.

## Notes
- The RAG pipeline relies on Qdrant, OpenAI, and Zep being reachable via
  environment variables. Missing keys will cause runtime errors.
