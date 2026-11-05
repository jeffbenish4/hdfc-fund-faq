# HDFC Fund Facts Assistant (RAG demo)
Facts-only MF FAQ bot. One citation per answer, <=3 sentences, "Last updated from sources" line, refuses advice/PII/returns.

## Architecture
`pipeline.py` (run once, locally/Colab): scrape -> load -> chunk -> embed (MiniLM) -> ChromaDB (+ export `data/store.json`).
Each stage output is saved: `data/raw`, `data/processed`, `data/chunks`, `data/embeddings`, `chroma_db/`.
Vercel runtime (`api/chat.py`): guardrails -> cosine retrieval on `store.json` -> Groq LLM. Query embedding = same MiniLM via HF Inference API.
Config: `config.yaml`. Secrets: `.env` (never commit).

## Run ingestion
    pip install -r requirements-ingest.txt
    python pipeline.py scrape    # then: load, chunk, embed, store  (or: python pipeline.py all)
Commit the generated `data/store.json`.

## Deploy on Vercel
Push to GitHub -> vercel.com -> Add New -> Project -> import repo -> Framework "Other" -> add env vars:
`GROQ_API_KEY`, `GROQ_MODEL` (a model id from console.groq.com/docs/models), `HF_TOKEN` (free HF token, "Inference Providers" permission) -> Deploy.

## Known limits
- Groww page structure can change; check `data/processed/*.json` facts after scraping.
- Source is Groww scheme pages (not AMC factsheets); "factsheet" links point to the scheme page.
- If the HF API fails, retrieval falls back to keyword matching.
- Answers are only as fresh as the last ingestion run.
