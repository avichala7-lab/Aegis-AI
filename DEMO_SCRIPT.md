# Aegis AI Demo Script

## 2-3 Minute Walkthrough

Hi, I built Aegis AI as a portfolio project to explore private, offline clinical document intelligence.

The problem I wanted to solve is that medical records are dense, sensitive, and hard to review quickly. A lot of AI tools are cloud-first, which is not ideal when working with privacy-sensitive healthcare documents. Aegis AI shows how a local AI stack can review a medical PDF, generate a structured patient dashboard, and answer questions with source evidence.

First, I open the app and upload a patient record PDF. The document is processed locally with PyMuPDF, split into chunks, embedded with Ollama using `nomic-embed-text`, and stored in a local ChromaDB vector database.

Next, I click **Generate Patient Summary**. The app creates a Patient Intelligence Dashboard with patient overview, active conditions, medications, allergies, abnormal labs, recent visit details, imaging findings, and risk flags. The goal is to make the document easier to scan, not to replace clinical judgment.

Then I move to the Q&A workspace. I can ask questions like:

- Summarize patient history
- List medications and allergies
- Identify abnormal findings
- What are the key risks?

When I ask a question, Aegis AI retrieves the most relevant document chunks and sends only that local context to Gemma running through Ollama. The answer appears in the chat, and I can expand the source chunks to inspect the evidence used.

To demonstrate the offline angle, I can turn WiFi off and continue using the app, as long as Ollama is running locally and the required models have already been downloaded.

The value proposition is simple: Aegis AI demonstrates an enterprise-style, privacy-focused AI workflow for sensitive documents using local RAG, local models, and source-grounded answers without relying on cloud model APIs.

This is an educational demo, not a medical device or clinical decision tool.
