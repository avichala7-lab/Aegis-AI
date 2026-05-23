import html
import json
import logging
import os
import re
import tempfile
from typing import List

import chromadb
import fitz
import ollama
import streamlit as st


EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3:4b"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_TOP_K = 8

SECTION_KEYWORDS = {
    "medications": [
        "medication",
        "medications",
        "current medications",
        "home medications",
        "prescription",
        "prescriptions",
        "drug",
        "drugs",
        "rx",
    ],
    "allergies": [
        "allergy",
        "allergies",
        "allergic",
        "adverse reaction",
        "adverse reactions",
    ],
    "labs": [
        "lab",
        "labs",
        "laboratory",
        "test",
        "tests",
        "cbc",
        "bmp",
        "cmp",
        "a1c",
        "hemoglobin",
        "creatinine",
        "glucose",
    ],
    "progress_notes": [
        "progress note",
        "progress notes",
        "subjective",
        "objective",
        "assessment",
        "plan",
    ],
    "discharge_summary": [
        "discharge summary",
        "hospital course",
        "discharge diagnosis",
        "discharge diagnoses",
        "discharge medications",
    ],
    "imaging": [
        "imaging",
        "radiology",
        "x-ray",
        "xray",
        "ct",
        "mri",
        "ultrasound",
    ],
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --surface: #f8f9fa;
            --surface-low: #f3f4f5;
            --surface-container: #edeeef;
            --card: #ffffff;
            --navy: #191c1d;
            --navy-soft: #414754;
            --blue: #005bbf;
            --blue-bright: #1a73e8;
            --blue-soft: #d8e2ff;
            --border: #c1c6d6;
            --text: #191c1d;
            --muted: #414754;
        }

        #MainMenu,
        header[data-testid="stHeader"],
        footer {
            visibility: hidden;
            height: 0;
        }

        html,
        body,
        .stApp {
            background: var(--surface);
            color: var(--text);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .block-container {
            max-width: 1440px;
            padding: 24px 40px 56px;
        }

        [data-testid="stSidebar"] {
            background: var(--surface-low);
            border-right: 1px solid var(--border);
            min-width: 280px;
            max-width: 280px;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 24px 20px;
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        h1,
        h2,
        h3,
        .font-display {
            font-family: "Public Sans", Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            letter-spacing: 0;
        }

        .clinical-hero,
        .feature-card,
        .demo-card,
        .workflow-step,
        .stack-strip,
        .document-card,
        .summary-dashboard-card,
        .disclaimer-card,
        .ai-card,
        .sidebar-card,
        .chat-panel {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 16px 36px rgba(0, 91, 191, 0.05);
        }

        .clinical-hero {
            padding: 34px 38px;
            margin-bottom: 18px;
            background: #ffffff;
        }

        .hero-topline {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 16px;
            align-items: start;
        }

        .clinical-hero h1 {
            margin: 0 0 8px;
            color: var(--text);
            font-size: 40px;
            line-height: 48px;
            font-weight: 700;
        }

        .clinical-hero p {
            margin: 0;
            color: var(--muted);
            font-size: 16px;
            line-height: 24px;
        }

        .hero-description {
            margin-top: 14px !important;
            max-width: 760px;
        }

        .security-badge,
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }

        .security-badge {
            gap: 8px;
            padding: 10px 14px;
            background: var(--surface-container);
            border: 1px solid var(--border);
            color: var(--text);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: nowrap;
        }

        .status-chip {
            padding: 6px 10px;
            background: var(--surface-low);
            border: 1px solid var(--border);
            color: var(--muted);
        }

        .status-chip.teal {
            background: var(--blue-soft);
            border-color: var(--blue-soft);
            color: var(--blue);
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 18px;
        }

        .section-label {
            margin: 28px 0 10px;
            color: var(--muted);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .feature-grid,
        .workflow-grid {
            display: grid;
            gap: 16px;
            margin: 18px 0;
        }

        .feature-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .workflow-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .feature-card,
        .demo-card,
        .workflow-step,
        .stack-strip,
        .document-card,
        .summary-dashboard-card,
        .disclaimer-card,
        .ai-card,
        .chat-panel {
            padding: 18px;
        }

        .feature-card h3,
        .demo-card h3,
        .workflow-step h3,
        .document-card h3,
        .summary-dashboard-card h3,
        .ai-card h3,
        .chat-panel h3 {
            margin: 0 0 8px;
            color: var(--text);
            font-size: 16px;
            line-height: 22px;
            font-weight: 700;
        }

        .feature-card p,
        .demo-card p,
        .workflow-step p,
        .document-card p,
        .summary-dashboard-card p,
        .disclaimer-card p,
        .ai-card p,
        .chat-panel p {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
            line-height: 20px;
        }

        .feature-icon {
            width: 36px;
            height: 36px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            background: var(--blue-soft);
            color: var(--blue);
            margin-bottom: 14px;
            font-size: 18px;
        }

        .demo-card {
            margin: 18px 0;
            padding: 20px;
        }

        .proof-list {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 8px;
            margin-top: 14px;
        }

        .workflow-step,
        .ai-card,
        .disclaimer-card {
            border-left: 4px solid var(--blue);
        }

        .stack-strip {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            margin: 18px 0;
            padding: 14px 16px;
        }

        .stack-item {
            color: var(--blue);
            background: var(--surface-low);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 13px;
            font-weight: 700;
        }

        .document-card {
            min-height: 224px;
        }

        .summary-dashboard-card {
            margin-top: 16px;
            padding: 18px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin-top: 14px;
        }

        .summary-list {
            margin: 8px 0 0;
            padding-left: 18px;
            color: var(--muted);
            font-size: 14px;
            line-height: 21px;
        }

        .lab-high {
            background: #fff1f2;
            color: #9f1239;
            border-color: #fecdd3;
        }

        .lab-low {
            background: #eff6ff;
            color: #1d4ed8;
            border-color: #bfdbfe;
        }

        .lab-abnormal {
            background: #fff7ed;
            color: #c2410c;
            border-color: #fed7aa;
        }

        .document-meta {
            display: grid;
            gap: 0;
            margin-top: 16px;
        }

        .meta-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 10px 0;
            border-top: 1px solid var(--border);
            color: var(--muted);
            font-size: 13px;
        }

        .meta-row strong {
            color: var(--text);
            text-align: right;
        }

        .chat-bubble {
            padding: 14px 16px;
            border-radius: 10px;
            margin: 12px 0;
            font-size: 14px;
            line-height: 20px;
        }

        .chat-bubble.user {
            background: var(--surface-container);
            border: 1px solid var(--border);
            margin-left: 10%;
        }

        .chat-bubble.assistant {
            background: #ffffff;
            border: 1px solid var(--border);
            border-left: 4px solid var(--blue);
            margin-right: 6%;
        }

        .bubble-label {
            display: block;
            margin-bottom: 6px;
            color: var(--blue);
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 28px;
        }

        .sidebar-logo {
            width: 42px;
            height: 42px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            background: var(--surface-container);
            color: var(--blue);
            font-size: 22px;
            border: 1px solid var(--border);
        }

        .sidebar-brand h2 {
            margin: 0;
            color: var(--text);
            font-size: 20px;
            line-height: 24px;
            font-weight: 800;
        }

        .sidebar-brand p {
            margin: 2px 0 0;
            color: var(--muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .sidebar-card {
            padding: 14px;
            margin-bottom: 14px;
            background: rgba(255, 255, 255, 0.62);
        }

        .sidebar-card h3 {
            margin: 0 0 10px;
            color: var(--muted);
            font-size: 12px;
            line-height: 16px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .sidebar-card p {
            margin: 0;
            color: var(--muted);
            font-size: 13px;
            line-height: 18px;
        }

        .stButton > button {
            min-height: 40px;
            border-radius: 8px;
            border: 1px solid var(--blue);
            background: #ffffff;
            color: var(--blue);
            font-weight: 700;
        }

        .stButton > button:hover {
            border-color: var(--navy);
            color: var(--navy);
            background: var(--surface-low);
        }

        .stButton > button[kind="primary"] {
            background: var(--blue);
            color: #ffffff;
            border-color: var(--blue);
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px dashed var(--border);
            border-radius: 10px;
            padding: 10px;
        }

        [data-testid="stChatInput"] {
            background: #ffffff;
            border-top: 1px solid var(--border);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: none;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 12px;
        }

        div[data-testid="stMetricLabel"] p {
            font-size: 12px;
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            font-size: 20px;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            font-size: 13px;
        }

        @media (max-width: 820px) {
            .block-container {
                padding: 16px;
            }

            [data-testid="stSidebar"] {
                min-width: auto;
                max-width: none;
            }

            .hero-topline,
            .feature-grid,
            .proof-list,
            .summary-grid,
            .workflow-grid {
                grid-template-columns: 1fr;
            }

            .security-badge {
                white-space: normal;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <section class="clinical-hero">
            <div class="hero-topline">
                <div>
                    <h1>Aegis AI</h1>
                    <p>Private clinical intelligence. Fully local.</p>
                    <p class="hero-description">
                        Upload medical records, generate structured patient intelligence dashboards,
                        and ask source-grounded questions — all without cloud APIs.
                    </p>
                </div>
                <div class="security-badge">Enterprise private AI</div>
            </div>
            <div class="badge-row">
                <span class="status-chip teal">Offline-first</span>
                <span class="status-chip">Private by design</span>
                <span class="status-chip">Source-grounded</span>
                <span class="status-chip">Local RAG</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    cta_one, cta_two, _ = st.columns([0.22, 0.24, 0.54])
    with cta_one:
        if st.button("Upload Patient Record", type="primary"):
            st.info("Use the sidebar upload control to select a patient PDF record.")
    with cta_two:
        if st.button("View Demo Workflow"):
            st.info("The demo workflow is shown below.")


def render_feature_cards():
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <h3>Private by Design</h3>
                <p>Patient documents stay on-device using local AI infrastructure.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">▦</div>
                <h3>Patient Intelligence Dashboard</h3>
                <p>Structured patient summaries with conditions, medications, allergies, labs, and risk flags.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⌕</div>
                <h3>Evidence-Based Clinical Q&A</h3>
                <p>Ask questions and inspect source evidence used for answers.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">◇</div>
                <h3>Offline Clinical Workspace</h3>
                <p>Built for secure AI experimentation and privacy-sensitive healthcare workflows.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_mode():
    st.markdown(
        """
        <div class="section-label">Demo Workflow</div>
        <div class="workflow-grid">
            <div class="workflow-step">
                <span class="status-chip teal">Step 1</span>
                <h3 style="margin-top: 12px;">Upload Record</h3>
                <p>Add one patient document using the sidebar upload control.</p>
            </div>
            <div class="workflow-step">
                <span class="status-chip">Step 2</span>
                <h3 style="margin-top: 12px;">Generate Dashboard</h3>
                <p>Create a structured local patient intelligence dashboard.</p>
            </div>
            <div class="workflow-step">
                <span class="status-chip">Step 3</span>
                <h3 style="margin-top: 12px;">Ask Clinical Questions</h3>
                <p>Query the document and inspect the evidence chunks used.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stack-strip">
            <span class="stack-item">Ollama</span>
            <span class="stack-item">Gemma</span>
            <span class="stack-item">ChromaDB</span>
            <span class="stack-item">Streamlit</span>
            <span class="stack-item">Local RAG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.markdown(
        """
        <div class="disclaimer-card">
            <p><strong>Educational demo only.</strong> Aegis AI is not medical advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_ollama_status() -> bool:
    """Check whether the local Ollama service is reachable."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def extract_text_from_pdf(pdf_file) -> str:
    """Save the uploaded PDF temporarily, then extract text with PyMuPDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_path = temp_file.name

    try:
        document = fitz.open(temp_path)
        pages = [page.get_text() for page in document]
        document.close()
        return "\n".join(pages)
    finally:
        os.remove(temp_path)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks so retrieval has useful context."""
    clean_text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def detect_chunk_sections(chunk: str) -> dict:
    """Tag a chunk with lightweight clinical sections when keywords are present."""
    lowered_chunk = chunk.lower()
    sections = {
        section: any(keyword in lowered_chunk for keyword in keywords)
        for section, keywords in SECTION_KEYWORDS.items()
    }

    primary_section = "general"
    for section in SECTION_KEYWORDS:
        if sections[section]:
            primary_section = section
            break

    return {
        "section": primary_section,
        "has_medications": sections["medications"],
        "has_allergies": sections["allergies"],
        "has_labs": sections["labs"],
        "has_progress_notes": sections["progress_notes"],
        "has_discharge_summary": sections["discharge_summary"],
        "has_imaging": sections["imaging"],
    }


def route_query_section(question: str) -> str:
    """Route simple extraction questions to the most relevant chunk section."""
    lowered_question = question.lower()
    route_matches = []

    if any(keyword in lowered_question for keyword in ["medication", "drug", "prescription"]):
        route_matches.append("medications")
    if "allergy" in lowered_question or "allergies" in lowered_question:
        route_matches.append("allergies")
    if any(keyword in lowered_question for keyword in ["lab", "blood", "test"]):
        route_matches.append("labs")

    if len(route_matches) == 1:
        return route_matches[0]

    return ""


def build_section_filter(section: str):
    """Create a Chroma metadata filter for routed extraction questions."""
    if section == "medications":
        return {
            "$and": [
                {"has_medications": True},
                {"has_allergies": False},
            ]
        }
    if section == "allergies":
        return {"has_allergies": True}
    if section == "labs":
        return {"has_labs": True}

    return None


def get_embedding(text: str) -> List[float]:
    """Ask Ollama for an embedding vector."""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def build_vector_store(chunks: List[str]):
    """Create a fresh local Chroma collection for the uploaded PDF."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_names = [
        collection if isinstance(collection, str) else collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    for index, chunk in enumerate(chunks):
        metadata = {
            "chunk_number": index + 1,
            **detect_chunk_sections(chunk),
        }
        collection.add(
            ids=[f"chunk-{index}"],
            embeddings=[get_embedding(chunk)],
            documents=[chunk],
            metadatas=[metadata],
        )

    return collection


def answer_question(question: str, source_chunks: List[str]) -> str:
    """Use the retrieved chunks as context for the local chat model."""
    context = "\n\n".join(
        f"Source chunk {index + 1}:\n{chunk}"
        for index, chunk in enumerate(source_chunks)
    )

    prompt = f"""
You are an offline clinical document review assistant.

Rules:
- Answer ONLY from the retrieved context below.
- If the retrieved context does not contain the answer, say exactly: "I do not know."
- Do not infer facts that are not present in the retrieved context.
- Never mix categories.
- When asked about medications, return exact medication names exactly as written in the context.
- When asked about medications, do not include allergies or allergy-only substances.
- When asked about medications, do not duplicate drug names.
- When asked about medications, preserve available dose, route, frequency, and timing exactly as written.
- When asked about allergies, list only allergies explicitly found in the context.
- When asked about abnormal lab findings, include only laboratory tests, lab values, and lab interpretations.
- Keep clinical observations, symptoms, diagnoses, imaging findings, and procedures separate from lab findings.
- If uncertain, say exactly: "I do not know."
- Mention when the context is incomplete or only partially answers the question.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def retrieve_source_chunks(
    question: str,
    collection,
    n_results: int = RETRIEVAL_TOP_K,
) -> List[str]:
    """Find the chunks that are most relevant to a question."""
    question_embedding = get_embedding(question)
    result_count = min(n_results, collection.count())
    routed_section = route_query_section(question)
    section_filter = build_section_filter(routed_section)

    if result_count == 0:
        return []

    query_args = {
        "query_embeddings": [question_embedding],
        "n_results": result_count,
    }

    if section_filter:
        query_args["where"] = section_filter

    results = collection.query(**query_args)
    source_chunks = results["documents"][0]

    logger.info(
        "Retrieved %s chunks for question: %s | route: %s",
        len(source_chunks),
        question,
        routed_section or "general",
    )
    for index, chunk in enumerate(source_chunks, start=1):
        logger.info("Retrieved chunk %s: %s", index, chunk[:500])

    return source_chunks


def ask_pdf_question(question: str) -> tuple[str, List[str]]:
    """Retrieve source chunks and answer a question about the processed PDF."""
    source_chunks = retrieve_source_chunks(
        question,
        st.session_state.collection,
        n_results=RETRIEVAL_TOP_K,
    )
    if not source_chunks:
        return "I do not know.", []

    answer = answer_question(question, source_chunks)
    return answer, source_chunks


def parse_json_response(response_text: str) -> dict:
    """Parse JSON from a model response, including fenced JSON blocks."""
    cleaned_response = response_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned_response, re.DOTALL)

    if fenced_match:
        cleaned_response = fenced_match.group(1).strip()
    else:
        json_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group(0)

    return json.loads(cleaned_response)


def parse_json_list_response(response_text: str) -> List:
    """Parse a JSON array from a model response, including fenced blocks."""
    cleaned_response = response_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned_response, re.DOTALL)

    if fenced_match:
        cleaned_response = fenced_match.group(1).strip()
    else:
        list_match = re.search(r"\[.*\]", cleaned_response, re.DOTALL)
        if list_match:
            cleaned_response = list_match.group(0)

    parsed = json.loads(cleaned_response)
    return parsed if isinstance(parsed, list) else []


def parse_bullet_list(response_text: str) -> List[str]:
    """Fallback parser for simple bullet or line-based model responses."""
    items = []
    for line in response_text.splitlines():
        cleaned_line = line.strip().lstrip("-*•0123456789. )").strip()
        if cleaned_line and cleaned_line.lower() not in {"not identified", "none"}:
            items.append(cleaned_line)
    return items


def parse_json_object_with_fallback(response_text: str, fallback: dict) -> dict:
    """Parse a JSON object; return fallback if the model returns imperfect JSON."""
    try:
        parsed = parse_json_response(response_text)
        return parsed if isinstance(parsed, dict) else fallback
    except (json.JSONDecodeError, TypeError):
        return fallback


def parse_json_list_with_fallback(response_text: str, fallback: List) -> List:
    """Parse a JSON list; fall back to bullets if the JSON is imperfect."""
    try:
        parsed = parse_json_list_response(response_text)
        if parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    bullets = parse_bullet_list(response_text)
    return bullets if bullets else fallback


def coerce_items_to_dicts(items: List, fields: List[str]) -> List[dict]:
    """Convert parsed JSON or bullet fallback items into table-friendly dictionaries."""
    coerced_items = []

    for item in items:
        if isinstance(item, dict):
            coerced_items.append(
                {field: clean_value(item.get(field)) for field in fields}
            )
        elif item:
            row = {field: "Not identified" for field in fields}
            row[fields[0]] = clean_value(item)
            coerced_items.append(row)

    return coerced_items


def default_structured_summary(error_message: str = "") -> dict:
    """Return an empty structured summary shape for missing or invalid JSON."""
    summary = {
        "patient_overview": {
            "full_name": "Not identified",
            "age": "Not identified",
            "gender": "Not identified",
            "dob": "Not identified",
            "patient_id": "Not identified",
            "emergency_contact": "Not identified",
        },
        "conditions": [],
        "medications": [],
        "allergies": [],
        "abnormal_labs": [],
        "imaging_findings": [],
        "recent_hospitalization_or_visit": {
            "reason": "No recent hospitalization or visit identified",
            "diagnosis": "No recent hospitalization or visit identified",
            "date": "No recent hospitalization or visit identified",
            "treatment": "No recent hospitalization or visit identified",
            "follow_up": "No recent hospitalization or visit identified",
        },
        "risk_flags": [],
    }

    if error_message:
        summary["parse_error"] = error_message

    return summary


def retrieve_structured_summary_context(collection) -> List[str]:
    """Retrieve section-specific context for the structured summary dashboard."""
    summary_queries = [
        "Patient demographics full name age gender DOB patient ID emergency contact",
        "Active conditions and diagnosed conditions",
        "Active medications exact medication names dosage frequency prescription drug",
        "Allergies exact allergy list",
        "Abnormal labs blood test values high low abnormal",
        "Recent hospitalization admission reason diagnosis admission date discharge date treatment summary",
        "Risk flags uncontrolled chronic condition kidney function cardiovascular follow-up concern",
    ]
    source_chunks = []
    seen_chunks = set()

    for query in summary_queries:
        for chunk in retrieve_source_chunks(query, collection, n_results=RETRIEVAL_TOP_K):
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                source_chunks.append(chunk)

    return source_chunks


def retrieve_context_for_summary(query: str) -> tuple[str, List[str]]:
    """Retrieve focused context for one dashboard section."""
    chunks = retrieve_source_chunks(
        query,
        st.session_state.collection,
        n_results=RETRIEVAL_TOP_K,
    )
    context = "\n\n".join(
        f"Source chunk {index + 1}:\n{chunk}"
        for index, chunk in enumerate(chunks)
    )
    return context, chunks


def ask_focused_extraction(query: str, prompt: str) -> tuple[str, List[str]]:
    """Run one small local extraction call against focused retrieved context."""
    context, chunks = retrieve_context_for_summary(query)

    if not chunks:
        return "Not identified", []

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}],
    )
    return response["message"]["content"], chunks


def extract_patient_overview() -> tuple[dict, List[str]]:
    fallback = default_structured_summary()["patient_overview"]
    response_text, chunks = ask_focused_extraction(
        "patient demographics name age gender DOB patient ID emergency contact",
        """
Use ONLY the provided context. Do not infer. If unknown, write "Not identified".
Return ONLY a JSON object:
{
  "full_name": "",
  "age": "",
  "gender": "",
  "dob": "",
  "patient_id": "",
  "emergency_contact": ""
}
""",
    )
    return parse_json_object_with_fallback(response_text, fallback), chunks


def extract_conditions() -> tuple[List[str], List[str]]:
    response_text, chunks = ask_focused_extraction(
        "diagnosed conditions active problems medical history assessment diagnosis",
        """
Use ONLY the provided context. Do not infer.
Extract diagnosed active conditions only.
If not found, return ["Not identified"].
Return ONLY a JSON array of strings.
""",
    )
    return parse_json_list_with_fallback(response_text, ["Not identified"]), chunks


def extract_medications() -> tuple[List[dict], List[str]]:
    response_text, chunks = ask_focused_extraction(
        "medication list current medications prescriptions dosage frequency",
        """
Use ONLY the provided context. Do not infer.
Extract medications only. Do NOT include allergies or allergy-only substances.
Preserve medication names, dosages, and frequencies exactly when available.
If not found, return [].
Return ONLY a JSON array:
[
  {"medication": "", "dosage": "", "frequency": "", "notes": ""}
]
""",
    )
    parsed = parse_json_list_with_fallback(response_text, [])
    return coerce_items_to_dicts(
        parsed,
        ["medication", "dosage", "frequency", "notes"],
    ), chunks


def extract_allergies() -> tuple[List[dict], List[str]]:
    response_text, chunks = ask_focused_extraction(
        "allergies adverse reactions allergy list",
        """
Use ONLY the provided context. Do not infer.
Extract allergies only.
If not found, return [].
Return ONLY a JSON array:
[
  {"allergen": "", "reaction": ""}
]
""",
    )
    parsed = parse_json_list_with_fallback(response_text, [])
    return coerce_items_to_dicts(parsed, ["allergen", "reaction"]), chunks


def extract_abnormal_labs() -> tuple[List[dict], List[str]]:
    response_text, chunks = ask_focused_extraction(
        "laboratory results blood urine panel abnormal high low WBC RBC hemoglobin platelets sodium potassium creatinine eGFR glucose HbA1c LDL HDL triglycerides CRP ESR",
        """
Use ONLY the provided context. Do not infer.
Return ONLY laboratory test results. Exclude vitals, imaging findings, symptoms, physical exam, and diagnoses.
Allowed examples include WBC, RBC, hemoglobin, platelets, sodium, potassium, creatinine, eGFR, glucose, HbA1c, LDL, HDL, triglycerides, CRP, ESR.
If not found, return [].
Return ONLY a JSON array:
[
  {"test": "", "value": "", "status": "", "context": ""}
]
""",
    )
    parsed = parse_json_list_with_fallback(response_text, [])
    return coerce_items_to_dicts(parsed, ["test", "value", "status", "context"]), chunks


def extract_imaging_findings() -> tuple[List[str], List[str]]:
    response_text, chunks = ask_focused_extraction(
        "imaging diagnostic findings CT X-ray MRI ultrasound radiology impression",
        """
Use ONLY the provided context. Do not infer.
Extract imaging or diagnostic findings only, including CT, X-ray, MRI, ultrasound, radiology findings or impressions.
Do not include laboratory values.
If not found, return ["Not identified"].
Return ONLY a JSON array of strings.
""",
    )
    return parse_json_list_with_fallback(response_text, ["Not identified"]), chunks


def extract_recent_visit() -> tuple[dict, List[str]]:
    fallback = default_structured_summary()["recent_hospitalization_or_visit"]
    response_text, chunks = ask_focused_extraction(
        "recent hospitalization visit admission discharge reason diagnosis treatment follow-up",
        """
Use ONLY the provided context. Do not infer.
Extract the most recent hospitalization or visit.
If not found, use "No recent hospitalization or visit identified" for every field.
Return ONLY a JSON object:
{
  "reason": "",
  "diagnosis": "",
  "date": "",
  "treatment": "",
  "follow_up": ""
}
""",
    )
    return parse_json_object_with_fallback(response_text, fallback), chunks


def extract_risk_flags() -> tuple[List[dict], List[str]]:
    response_text, chunks = ask_focused_extraction(
        "risk flags uncontrolled chronic condition abnormal kidney function cardiovascular missing follow-up concern",
        """
Use ONLY the provided context. Do not infer or hallucinate.
Create risk flags only when supported by the document context.
If not found, return [].
Return ONLY a JSON array:
[
  {"risk": "", "why_it_matters": "", "evidence": ""}
]
""",
    )
    parsed = parse_json_list_with_fallback(response_text, [])
    return coerce_items_to_dicts(
        parsed,
        ["risk", "why_it_matters", "evidence"],
    ), chunks


def generate_structured_summary() -> tuple[dict, List[str], str]:
    """Build the dashboard from focused local extraction calls."""
    all_source_chunks = []

    overview, chunks = extract_patient_overview()
    all_source_chunks.extend(chunks)
    conditions, chunks = extract_conditions()
    all_source_chunks.extend(chunks)
    medications, chunks = extract_medications()
    all_source_chunks.extend(chunks)
    allergies, chunks = extract_allergies()
    all_source_chunks.extend(chunks)
    abnormal_labs, chunks = extract_abnormal_labs()
    all_source_chunks.extend(chunks)
    imaging_findings, chunks = extract_imaging_findings()
    all_source_chunks.extend(chunks)
    recent_visit, chunks = extract_recent_visit()
    all_source_chunks.extend(chunks)
    risk_flags, chunks = extract_risk_flags()
    all_source_chunks.extend(chunks)

    summary = {
        "patient_overview": overview,
        "conditions": conditions,
        "medications": medications,
        "allergies": allergies,
        "abnormal_labs": abnormal_labs,
        "imaging_findings": imaging_findings,
        "recent_hospitalization_or_visit": recent_visit,
        "risk_flags": risk_flags,
    }

    deduped_chunks = list(dict.fromkeys(all_source_chunks))
    raw_response = json.dumps(summary)
    return summary, deduped_chunks, raw_response


def generate_structured_patient_summary() -> tuple[dict, List[str], str]:
    """Backward-compatible wrapper for the structured summary generator."""
    return generate_structured_summary()


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "patient_summary" not in st.session_state:
        st.session_state.patient_summary = ""
    if "structured_summary" not in st.session_state:
        st.session_state.structured_summary = {}
    if "document_status" not in st.session_state:
        st.session_state.document_status = "Waiting for upload"


def clear_processed_document(document_status: str):
    """Clear UI state tied to a previously processed PDF."""
    for key in [
        "collection",
        "chunks",
        "document_name",
        "messages",
        "patient_summary",
        "structured_summary",
        "structured_summary_raw",
        "summary_sources",
    ]:
        st.session_state.pop(key, None)

    st.session_state.messages = []
    st.session_state.patient_summary = ""
    st.session_state.structured_summary = {}
    st.session_state.document_status = document_status


def render_sidebar():
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">✚</div>
            <div>
                <h2>Aegis AI</h2>
                <p>Clinical Intelligence Workspace</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_ready = check_ollama_status()
    model_label = "Online locally" if model_ready else "Not reachable"
    model_chip = "teal" if model_ready else ""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <h3>Model status</h3>
            <span class="status-chip {model_chip}">{model_label}</span>
            <p style="margin-top: 10px;"><strong>Active model</strong><br>{CHAT_MODEL}</p>
            <p style="margin-top: 8px;"><strong>Embeddings</strong><br>{EMBEDDING_MODEL}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model_ready:
        st.sidebar.success("Ollama is reachable")
    else:
        st.sidebar.error("Ollama is not reachable")

    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <h3>Upload PDF</h3>
            <p>Select one medical document for local review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_pdf = st.sidebar.file_uploader("Choose one PDF", type=["pdf"])

    if uploaded_pdf is not None:
        upload_size = getattr(uploaded_pdf, "size", None)
        selected_document = f"{uploaded_pdf.name}:{upload_size}"

        if st.session_state.get("selected_document") != selected_document:
            clear_processed_document("PDF selected - ready to process")
            st.session_state.selected_document = selected_document
    else:
        if st.session_state.get("selected_document") is not None:
            clear_processed_document("Waiting for upload")
        st.session_state.selected_document = None

    document_ready = "chunks" in st.session_state
    document_chip = "teal" if document_ready else ""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <h3>Processing state</h3>
            <span class="status-chip {document_chip}">{st.session_state.document_status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chunks" in st.session_state:
        st.sidebar.metric("Chunks indexed", len(st.session_state.chunks))

    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <h3>Privacy disclaimer</h3>
            <p>Files are processed locally for this MVP. No external web APIs are used by the app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return uploaded_pdf


def process_uploaded_pdf(uploaded_pdf):
    with st.spinner("Extracting text and building the local vector store..."):
        clear_processed_document("Processing PDF")
        text = extract_text_from_pdf(uploaded_pdf)
        chunks = chunk_text(text)

        if not chunks:
            st.session_state.document_status = "No text found"
            st.error("No text was found in this PDF.")
            return

        st.session_state.collection = build_vector_store(chunks)
        st.session_state.chunks = chunks
        st.session_state.document_name = uploaded_pdf.name
        st.session_state.messages = []
        st.session_state.patient_summary = ""
        st.session_state.structured_summary = {}
        st.session_state.document_status = f"Processed {len(chunks)} chunks"

    st.success(f"PDF processed into {len(chunks)} chunks.")


def render_workflow():
    st.markdown(
        """
        <div class="section-label">Workflow</div>
        <div class="workflow-grid">
            <div class="workflow-step">
                <span class="status-chip teal">Step 1</span>
                <h3 style="margin-top: 12px;">Upload PDF</h3>
                <p>Select one document in the sidebar.</p>
            </div>
            <div class="workflow-step">
                <span class="status-chip">Step 2</span>
                <h3 style="margin-top: 12px;">Generate patient summary</h3>
                <p>Create a concise AI-assisted review from local context.</p>
            </div>
            <div class="workflow-step">
                <span class="status-chip">Step 3</span>
                <h3 style="margin-top: 12px;">Ask questions</h3>
                <p>Chat with source-grounded answers and expandable evidence.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_card(uploaded_pdf):
    document_name = st.session_state.get("document_name")
    if uploaded_pdf is not None and not document_name:
        document_name = uploaded_pdf.name

    document_label = document_name or "No PDF selected"
    chunk_label = len(st.session_state.chunks) if "chunks" in st.session_state else 0
    ready_label = "Ready for questions" if "collection" in st.session_state else "Awaiting processing"
    ready_chip = "teal" if "collection" in st.session_state else ""

    st.markdown(
        f"""
        <div class="document-card">
            <span class="status-chip {ready_chip}">{ready_label}</span>
            <h3 style="margin-top: 14px;">Patient document</h3>
            <p>{html.escape(document_label)}</p>
            <div class="document-meta">
                <div class="meta-row"><span>Storage</span><strong>Local ChromaDB</strong></div>
                <div class="meta-row"><span>Chunks indexed</span><strong>{chunk_label}</strong></div>
                <div class="meta-row"><span>Answer model</span><strong>{CHAT_MODEL}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    process_disabled = uploaded_pdf is None
    if st.button("Process uploaded PDF", disabled=process_disabled, type="primary"):
        process_uploaded_pdf(uploaded_pdf)

    summary_disabled = "collection" not in st.session_state
    if st.button("Generate Patient Summary", disabled=summary_disabled):
        with st.spinner("Generating patient summary from the PDF..."):
            structured_summary, source_chunks, raw_response = generate_structured_summary()
            st.session_state.structured_summary = structured_summary
            st.session_state.structured_summary_raw = raw_response
            st.session_state.summary_sources = source_chunks

    if st.session_state.structured_summary:
        render_structured_summary_dashboard(st.session_state.structured_summary)
        if st.session_state.get("summary_sources"):
            with st.expander("Optional evidence: source chunks used for structured summary"):
                for index, chunk in enumerate(st.session_state.summary_sources, start=1):
                    st.markdown(f"**Source chunk {index}**")
                    st.write(chunk)


def clean_value(value, fallback: str = "Not identified") -> str:
    """Return a clean display value for missing model fields."""
    if value is None:
        return fallback
    value = str(value).strip()
    return value if value else fallback


def normalize_string_list(value) -> List[str]:
    """Normalize model values into a list of strings."""
    if isinstance(value, list):
        items = [clean_value(item) for item in value if clean_value(item)]
        return items or ["Not identified"]
    return [clean_value(value)]


def normalize_dict_list(value) -> List[dict]:
    """Normalize model values into a list of dictionaries."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def count_identified_list(value) -> int:
    """Count useful entries in a summary list."""
    if not isinstance(value, list):
        return 0
    return len(
        [
            item
            for item in value
            if clean_value(item) != "Not identified"
            if not isinstance(item, dict) or any(
                clean_value(field) != "Not identified" for field in item.values()
            )
        ]
    )


def infer_risk_level(summary: dict) -> str:
    """Small display-only risk label based on extracted flags and abnormal labs."""
    risk_count = count_identified_list(summary.get("risk_flags", []))
    lab_count = count_identified_list(summary.get("abnormal_labs", []))

    if risk_count >= 2 or lab_count >= 3:
        return "High"
    if risk_count or lab_count:
        return "Review"
    return "Not identified"


def render_patient_overview_card(overview: dict):
    if not isinstance(overview, dict):
        overview = {}

    with st.container(border=True):
        st.caption("Patient Snapshot")
        rows = [
            {"Field": "Full name", "Value": clean_value(overview.get("full_name"))},
            {"Field": "Age", "Value": clean_value(overview.get("age"))},
            {"Field": "Gender", "Value": clean_value(overview.get("gender"))},
            {"Field": "DOB", "Value": clean_value(overview.get("dob"))},
            {"Field": "Patient ID", "Value": clean_value(overview.get("patient_id"))},
            {"Field": "Emergency contact", "Value": clean_value(overview.get("emergency_contact"))},
        ]
        st.table(rows)


def render_conditions_card(conditions):
    with st.container(border=True):
        st.caption("Active Conditions")
        items = [item for item in normalize_string_list(conditions) if item != "Not identified"]

        if items:
            cols = st.columns(2)
            for index, item in enumerate(items):
                with cols[index % 2]:
                    st.success(item)
        else:
            st.info("Not identified in document.")


def render_medications_card(medications):
    rows = []
    for medication in normalize_dict_list(medications):
        rows.append(
            {
                "Medication": clean_value(medication.get("medication")),
                "Dosage": clean_value(medication.get("dosage")),
                "Frequency": clean_value(medication.get("frequency")),
                "Notes": clean_value(medication.get("notes")),
            }
        )

    with st.container(border=True):
        st.caption("Medications")

        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Not identified in document.")


def render_allergies_card(allergies):
    rows = []
    for allergy in normalize_dict_list(allergies):
        rows.append(
            {
                "Allergen": clean_value(allergy.get("allergen")),
                "Reaction": clean_value(allergy.get("reaction")),
            }
        )

    with st.container(border=True):
        st.caption("Allergies")

        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Not identified in document.")


def render_abnormal_labs_card(labs):
    rows = []
    for lab in normalize_dict_list(labs):
        rows.append(
            {
                "Test": clean_value(lab.get("test")),
                "Value": clean_value(lab.get("value")),
                "Status": clean_value(lab.get("status")),
                "Context": clean_value(lab.get("context")),
            }
        )

    with st.container(border=True):
        st.caption("Abnormal Labs")

        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Not identified in document.")


def render_imaging_findings_card(findings):
    with st.expander("Imaging Findings", expanded=False):
        items = [item for item in normalize_string_list(findings) if item != "Not identified"]

        if items:
            for item in items:
                st.info(item)
        else:
            st.info("Not identified in document.")


def render_visit_card(visit: dict):
    if not isinstance(visit, dict):
        visit = {}
    fallback = "No recent hospitalization or visit identified"

    with st.container(border=True):
        st.caption("Recent Visit / Hospitalization")
        rows = [
            {"Field": "Reason", "Value": clean_value(visit.get("reason"), fallback)},
            {"Field": "Diagnosis", "Value": clean_value(visit.get("diagnosis"), fallback)},
            {"Field": "Date", "Value": clean_value(visit.get("date"), fallback)},
            {"Field": "Treatment", "Value": clean_value(visit.get("treatment"), fallback)},
            {"Field": "Follow-up", "Value": clean_value(visit.get("follow_up"), fallback)},
        ]
        st.table(rows)


def render_risk_flags_card(risk_flags):
    flags = normalize_dict_list(risk_flags)

    with st.container(border=True):
        st.caption("Risk Flags")

        if not flags:
            st.info("Not identified in document.")
            return

        for flag in flags:
            with st.container(border=True):
                st.warning(clean_value(flag.get("risk")))
                st.caption(f"Why: {clean_value(flag.get('why_it_matters'))}")
                st.caption(f"Severity: {clean_value(flag.get('severity'), 'Review')}")


def render_summary_metric_cards(summary: dict):
    overview = summary.get("patient_overview", {})
    if not isinstance(overview, dict):
        overview = {}
    patient = clean_value(overview.get("full_name"))
    metrics = [
        ("Patient Name", patient),
        ("Risk Level", infer_risk_level(summary)),
        ("Conditions Count", str(count_identified_list(summary.get("conditions", [])))),
        ("Abnormal Labs Count", str(count_identified_list(summary.get("abnormal_labs", [])))),
    ]
    columns = st.columns(4)

    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)


def render_structured_summary_dashboard(summary: dict):
    st.subheader("Patient Intelligence Dashboard")
    st.caption("AI-assisted structured summary generated from the uploaded document.")

    render_summary_metric_cards(summary)

    left_column, right_column = st.columns([0.7, 0.3], gap="medium")
    with left_column:
        render_patient_overview_card(summary.get("patient_overview", {}))
        render_visit_card(summary.get("recent_hospitalization_or_visit", {}))
        render_conditions_card(summary.get("conditions", []))
        render_medications_card(summary.get("medications", []))
        render_abnormal_labs_card(summary.get("abnormal_labs", []))

    with right_column:
        render_risk_flags_card(summary.get("risk_flags", []))
        render_allergies_card(summary.get("allergies", []))
        render_imaging_findings_card(summary.get("imaging_findings", []))

    if summary.get("parse_error"):
        st.warning("The model response was not valid JSON, so a fallback dashboard was shown.")


def handle_user_question(question: str) -> bool:
    """Handle one user question from either chat input or an example prompt."""
    clean_question = question.strip()
    if not clean_question:
        return False

    if "collection" not in st.session_state:
        st.error("Upload and process a PDF before asking questions.")
        return False

    st.session_state.messages.append({"role": "user", "content": clean_question})

    with st.spinner("Aegis AI is reviewing local context..."):
        answer, source_chunks = ask_pdf_question(clean_question)

    if not answer or not answer.strip():
        st.error("No answer was generated. Please try again.")
        return False

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_chunks,
        }
    )
    st.rerun()
    return True


def render_example_prompts() -> str:
    st.markdown(
        '<div class="section-label" style="margin-top: 16px;">Example prompts</div>',
        unsafe_allow_html=True,
    )
    prompts = [
        "Summarize patient history",
        "List medications and allergies",
        "Identify abnormal findings",
        "What are the key risks?",
    ]
    columns = st.columns(2)

    for index, prompt in enumerate(prompts):
        with columns[index % 2]:
            if st.button(
                prompt,
                key=f"example_prompt_{index}",
                disabled="collection" not in st.session_state,
                use_container_width=True,
            ):
                return prompt

    return ""


def render_message(message):
    label = "You" if message["role"] == "user" else "Local AI"
    bubble_class = "user" if message["role"] == "user" else "assistant"
    safe_content = html.escape(message["content"]).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="chat-bubble {bubble_class}">
            <span class="bubble-label">{label}</span>
            {safe_content}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if message["role"] == "assistant" and message.get("sources"):
        with st.expander("Source chunks used"):
            for index, chunk in enumerate(message["sources"], start=1):
                st.markdown(f"**Source chunk {index}**")
                st.write(chunk)


def render_chat_interface():
    st.markdown(
        """
        <div class="chat-panel">
            <span class="status-chip teal">Q&A workspace</span>
            <h3 style="margin-top: 12px;">Ask questions</h3>
            <p>Ask focused questions and review the source chunks used for each answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_prompt = render_example_prompts()

    for message in st.session_state.messages:
        render_message(message)

    typed_question = st.chat_input(
        "Ask a question about the PDF",
        disabled="collection" not in st.session_state,
    )
    question = selected_prompt or typed_question

    if question:
        handle_user_question(question)


def render_clinical_workspace(uploaded_pdf):
    st.markdown('<div class="section-label">Main clinical workspace</div>', unsafe_allow_html=True)
    left_column, right_column = st.columns([0.95, 1.25], gap="large")

    with left_column:
        render_document_card(uploaded_pdf)

    with right_column:
        render_chat_interface()


def main():
    st.set_page_config(
        page_title="Aegis AI",
        page_icon="🏥",
        layout="wide",
    )
    initialize_session_state()
    apply_custom_css()

    uploaded_pdf = render_sidebar()
    render_hero()
    render_feature_cards()
    render_demo_mode()
    render_clinical_workspace(uploaded_pdf)
    render_disclaimer()


if __name__ == "__main__":
    main()
