import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart RAG Q&A System",
    page_icon="📘",
    layout="wide"
)

# -----------------------------
# Custom CSS for Professional UI
# -----------------------------
st.markdown("""
<style>

/* Main app background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #020617 60%);
    color: #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid #1e293b;
}

/* Headings */
h1, h2, h3 {
    color: #f8fafc;
}

/* Text inputs */
input, textarea {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border: 1px solid #334155 !important;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: 600;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
}

/* Chat bubbles */
.chat-user {
    background-color: #1e293b;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 6px;
}

.chat-bot {
    background-color: #020617;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    border-left: 4px solid #3b82f6;
}

/* Source boxes */
.source-box {
    background-color: #020617;
    padding: 10px;
    border-radius: 10px;
    font-size: 14px;
    margin-bottom: 6px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)



# -----------------------------
# Header
# -----------------------------
st.markdown("## 🤖 Smart Document Question Answering System (RAG)")
st.write(
    "Upload one or more documents and ask questions. "
    "Answers are generated using **Retrieval-Augmented Generation** and are always grounded in the source content."
)
st.divider()

# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# -----------------------------
# Load Embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -----------------------------
# Load LLM
# -----------------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )

llm = load_llm()

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write(
        "This is a **generic RAG-based Question Answering system**. "
        "It works for legal, academic, technical, and general documents by "
        "retrieving relevant content and generating grounded answers."
    )

# =============================
# Document Processing
# =============================
if uploaded_files:
    with st.spinner("📚 Processing documents and building knowledge base..."):
        all_documents = []

        for uploaded_file in uploaded_files:
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Store document name in metadata
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(all_documents)

        st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)

    st.success("✅ Documents indexed successfully!")

# =============================
# Chat Interface
# =============================
if st.session_state.vector_db:
    st.markdown("### 💬 Ask Questions")

    user_query = st.text_input(
        "Enter your question",
        placeholder="e.g., What are the key objectives mentioned in the document?"
    )

    if user_query:
        docs = st.session_state.vector_db.similarity_search(user_query, k=4)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful and knowledgeable assistant.

Answer the user’s question clearly in 1–2 complete sentences using ONLY the information provided in the context below.
Do NOT use outside knowledge.
Do NOT repeat the question.
Do NOT invent information.

If the question asks for a specific value (such as a name, title, date, or number) and it is NOT explicitly mentioned in the context, respond with:
"The provided context does not contain enough information."

If the answer contains multiple points or types, list ALL of them clearly.

Context:
{context}

Question:
{user_query}

Answer:
"""

        with st.spinner("🤖 Generating answer..."):
            answer = llm(prompt)[0]["generated_text"]

        # Prepare source citations
        sources = []
        for doc in docs:
            source_name = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "N/A")
            snippet = doc.page_content[:200] + "..."
            sources.append((source_name, page, snippet))

        st.session_state.chat_history.append(
            {
                "user": user_query,
                "bot": answer,
                "sources": sources
            }
        )

# =============================
# Display Chat History + Sources
# =============================
if st.session_state.chat_history:
    st.markdown("### 📝 Conversation")

    for chat in st.session_state.chat_history:
        st.markdown(
            f"<div class='chat-user'><b>🧑 User:</b> {chat['user']}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='chat-bot'><b>🤖 Answer:</b> {chat['bot']}</div>",
            unsafe_allow_html=True
        )

        with st.expander("📌 View Sources"):
            for src in chat["sources"]:
                st.markdown(
                    f"<div class='source-box'>"
                    f"<b>📄 {src[0]}</b> | Page {src[1]}<br>"
                    f"<i>{src[2]}</i>"
                    f"</div>",
                    unsafe_allow_html=True
                )
