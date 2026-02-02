from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Load embeddings (same as ingest.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# User query
query = input("Ask a question: ")

# Retrieve relevant chunks
docs = vector_db.similarity_search(query, k=3)

context = "\n".join([doc.page_content for doc in docs])

# Load instruction-following model (VERY IMPORTANT)
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

prompt = f"""
You are a helpful and knowledgeable assistant.

Answer the user’s question clearly in 1–2 complete sentences using ONLY the information provided in the context below.
Do NOT use outside knowledge.
Do NOT repeat the question.
Do NOT invent information.

If the answer contains multiple points or types, list ALL of them clearly.
If the context is insufficient, say: "The provided context does not contain enough information."

Context:
{context}

Question:
{query}

Answer:

"""

result = qa_pipeline(prompt)[0]["generated_text"]

print("\n--- Final Answer ---\n")
print(result)
