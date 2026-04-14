# MedQA

**Production Medical Question Answering Pipeline**

A modular NLP system that classifies medical questions, extracts symptoms, retrieves relevant context, and generates safe, transparent answers.

Author: Vivek Vasisht Ediga

---

## Architecture

The pipeline processes questions through four stages:

| Stage | Model | Purpose |
|-------|-------|---------|
| Intent Classification | Llama-3.3-70B (zero-shot) | Classify into 10 medical intent categories |
| Symptom Extraction | Llama-3.1-8B (few-shot) | Extract symptom, body location, duration, trigger |
| Semantic Retrieval | MiniLM-L6-v2 + FAISS | Retrieve similar questions from 3,173-question corpus |
| Answer Generation | Llama-3.1-8B | Generate context-augmented medical response |

## Project Structure

```
MedQA/
├── src/medqa/              # Core Python package
│   ├── config.py           # Pydantic settings + YAML config
│   ├── log.py              # Structured logging
│   ├── models/             # Schemas, LLM client, embeddings
│   ├── pipeline/           # Intent, symptoms, retrieval, generation, orchestrator
│   ├── data/               # Data loader, FAISS vector store
│   ├── evaluation/         # Metrics, baselines, evaluator
│   └── api/                # FastAPI REST API
├── frontend/               # Streamlit multi-page dashboard
│   ├── app.py              # Home page
│   ├── pages/              # Analysis, Evaluation, Explorer
│   └── static/             # Custom CSS
├── tests/                  # pytest test suite
├── scripts/                # CLI tools (evaluate, build_index)
├── configs/                # YAML configuration
├── data/                   # Corpus + labeled data
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # API + Frontend services
└── Makefile                # Common commands
```

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# 3. Run the frontend
make frontend

# 4. Or run the API
make api
```

## Commands

| Command | Description |
|---------|-------------|
| `make install` | Install the package |
| `make dev` | Install with dev dependencies |
| `make frontend` | Launch Streamlit dashboard |
| `make api` | Launch FastAPI server |
| `make test` | Run test suite |
| `make lint` | Run linter |
| `make evaluate` | Run evaluation suite |
| `make docker-up` | Build and start Docker containers |

## API

```
POST /api/v1/analyze
{
  "question": "Why does my chest feel tight?",
  "top_k": 5
}

GET /api/v1/health
```

## Evaluation

| Model | Accuracy |
|-------|----------|
| Rule-Based | 35% |
| TF-IDF + Random Forest | 49% |
| TF-IDF + Logistic Regression | 49% |
| **LLM (Llama-3.3-70B)** | **77%** |

## Tech Stack

- **LLM Inference:** Groq API (Llama 3.x)
- **Embeddings:** sentence-transformers (MiniLM-L6-v2)
- **Vector Search:** FAISS
- **API:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Validation:** Pydantic
- **Testing:** pytest
- **Containerization:** Docker