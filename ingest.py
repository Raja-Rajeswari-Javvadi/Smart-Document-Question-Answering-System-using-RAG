from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# HuggingFace embeddings (FREE, LOCAL)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector DB
vector_db = FAISS.from_documents(chunks, embeddings)

# Save vector DB
vector_db.save_local("faiss_index")

print("Document successfully indexed using HuggingFace embeddings!")
