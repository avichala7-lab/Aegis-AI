# Aegis AI Project Summary

## What I Built

Aegis AI is an offline-first clinical intelligence workspace built with Python and Streamlit. It allows a user to upload one medical PDF, generate a structured patient dashboard, and ask source-grounded questions about the document using local AI models.

The app uses PyMuPDF for PDF extraction, ChromaDB for local vector search, Ollama with `nomic-embed-text` for embeddings, and Ollama with `gemma3:4b` for local answer generation.

## Why Offline AI Matters

Healthcare documents can contain sensitive personal and clinical information. Cloud AI workflows may be difficult to use in privacy-sensitive environments unless strong controls are in place.

Aegis AI demonstrates a local-first pattern where document parsing, embeddings, retrieval, and answer generation happen on the user's machine. This makes the project relevant to enterprise AI scenarios where privacy, security, auditability, and data control matter.

## Design Decisions

- **Streamlit for speed:** chosen to build a functional portfolio MVP quickly with a clear interactive workflow.
- **Ollama for local models:** avoids external model APIs and supports offline inference after model download.
- **ChromaDB for local retrieval:** keeps vector storage simple and local.
- **Source-grounded Q&A:** answers are generated from retrieved document chunks and source chunks are shown to the user.
- **Structured dashboard:** turns dense clinical text into scan-friendly sections such as medications, allergies, labs, visits, and risk flags.
- **Simple codebase:** the project is intentionally compact and readable for demonstration and learning.

## Limitations

- Processes one PDF at a time.
- Not a medical device and not intended for diagnosis or treatment.
- Extraction quality depends on PDF text quality and local model performance.
- Does not include authentication, audit logs, encryption, or role-based access control.
- Does not handle scanned PDFs that require OCR.
- Local model responses may still require human review.

## Next Steps

- Add OCR support for scanned documents.
- Add stronger structured extraction validation.
- Add evaluation tests with sample synthetic records.
- Add multi-document patient timelines.
- Add audit logs and security controls for enterprise environments.
- Add configurable local model selection.
- Improve dashboard export and reporting.

## Relevance To Enterprise AI Builder Roles

Aegis AI demonstrates the ability to design and build a practical AI application around sensitive data constraints. It combines local LLM infrastructure, retrieval-augmented generation, structured extraction, user-centered product design, and a clear privacy story.

This type of project is relevant to enterprise AI builder roles because it shows how to move beyond a basic chatbot and create a focused workflow that addresses trust, source evidence, usability, and local deployment constraints.
