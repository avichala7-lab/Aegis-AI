# Offline Medical Copilot - MVP 1 Record

## Project Goal

Build a beginner-friendly offline medical PDF question-answering app.

This MVP focuses only on the first working version:

- Upload one PDF.
- Extract text from the PDF.
- Split the text into chunks.
- Create embeddings locally with Ollama.
- Store chunks locally in ChromaDB.
- Ask questions about the uploaded PDF.
- Answer using a local Ollama model.
- Show the source chunks used for each answer.

## Current Folder

Project workspace:

```text
/Users/awichalabhishek/Documents/offline-medical-copilot
```

## Files Created

### `app.py`

Main Streamlit application.

It contains:

- PDF upload UI.
- PDF text extraction using PyMuPDF.
- Simple text chunking.
- Local embeddings using Ollama `nomic-embed-text`.
- Local vector storage using ChromaDB.
- Question answering using Ollama `gemma3:4b`.
- Source chunk display under each answer.

### `requirements.txt`

Python dependencies for the MVP:

```text
streamlit
PyMuPDF
ollama
chromadb
```

### `README.md`

Project instructions.

It explains:

- What the app does.
- How the MVP works.
- How to install dependencies.
- How to pull Ollama models.
- How to run Streamlit.

### `.gitignore`

Keeps generated local files out of version control:

```text
__pycache__/
*.pyc
chroma_db/
```

## Local Models Used

Embedding model:

```bash
ollama pull nomic-embed-text
```

Answer model:

```bash
ollama pull gemma3:4b
```

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Ollama:

```bash
ollama serve
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The app runs at:

```text
http://127.0.0.1:8501
```

## MVP Flow

1. User uploads one PDF.
2. App saves it temporarily.
3. PyMuPDF extracts text from all pages.
4. Text is cleaned and split into overlapping chunks.
5. Each chunk is embedded with `nomic-embed-text`.
6. Chunks and embeddings are stored in local ChromaDB.
7. User asks a question.
8. The question is embedded.
9. ChromaDB retrieves the most relevant chunks.
10. `gemma3:4b` answers using only those chunks.
11. The app displays the answer and the source chunks.

## Features Intentionally Not Added Yet

These were deliberately left out to keep MVP 1 simple:

- Patient profiles.
- Voice input or output.
- SQLite database.
- Authentication.
- Multi-document upload.
- Advanced medical workflows.
- Complex UI.
- Cloud APIs.

## Verification Completed

Completed checks:

- Python syntax check passed.
- Required Python packages installed.
- Imports verified for Streamlit, PyMuPDF, Ollama, and ChromaDB.
- Streamlit server started successfully.
- Local server responded with HTTP 200.

## Learning Notes

This MVP is useful because it teaches the core Retrieval-Augmented Generation flow:

```text
PDF -> Text -> Chunks -> Embeddings -> Vector Store -> Retrieval -> LLM Answer -> Sources
```

The important idea is that the language model does not read the whole PDF at answer time.
Instead, the app searches for the most relevant chunks first, then gives only those chunks to the model as context.

## Next Sensible Step

For MVP 2, a good next step would be improving reliability before adding features:

- Better chunking by page number.
- Show page numbers with source chunks.
- Add clearer errors if Ollama is not running.
- Add a reset button.
- Add basic tests for chunking.
