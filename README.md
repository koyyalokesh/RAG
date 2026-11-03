# RAG Learning Project

A small retrieval-augmented generation (RAG) project built with LangChain, OpenAI embeddings, and Chroma. It demonstrates the core stages of a RAG pipeline using customer-support policy documents.

## What This Project Covers

1. Loading and splitting Markdown documents into chunks
2. Creating embeddings and storing them in Chroma
3. Comparing similarity, filtered, MMR, and score-threshold retrieval
4. Building an end-to-end RAG chain that answers questions from retrieved context

## Project Structure

```text
.
├── data/                       # Support policy documents
├── src/rag/                    # Python package
├── 01_splitter.py              # Document loading and chunking
├── 02_chromstore.py            # Persistent Chroma vector store
├── 03_retreival-strategy.py    # Retrieval strategy examples
├── 04_basic_ragflow.py         # End-to-end RAG question answering
├── pyproject.toml              # Project metadata and dependencies
└── uv.lock                     # Locked dependency versions
```

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key

## Setup

Install the dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is ignored by Git and must not be committed.

## Run the Examples

Run each example from the project root in order:

```bash
uv run python 01_splitter.py
uv run python 02_chromstore.py
uv run python 03_retreival-strategy.py
uv run python 04_basic_ragflow.py
```

The scripts use the Markdown files in `data/` as their knowledge base. Scripts that persist embeddings create a local `chroma_store/` directory, which is excluded from version control because it is generated data.

## Sample Knowledge Base

The included documents cover:

- Refunds and cancellations
- Shipping and delivery
- Product warranties

The final example answers supported questions from this content and declines questions that are outside the available context.
