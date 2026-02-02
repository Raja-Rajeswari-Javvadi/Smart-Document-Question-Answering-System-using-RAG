# 📄 Smart Document Question Answering System using RAG

A **Retrieval-Augmented Generation (RAG)** based application that allows users to upload documents and ask natural language questions, providing accurate and context-aware answers by combining semantic search with Large Language Models (LLMs).

---

## 🧠 Project Overview

Traditional question-answering systems rely only on pre-trained knowledge, which limits their ability to answer questions from custom documents.  
This project overcomes that limitation by using **RAG**, where relevant document chunks are retrieved first and then passed to an LLM to generate precise answers.

The system is suitable for:
- Academic projects
- Intelligent document search
- Knowledge base question answering
- AI-powered assistants

---

## 🚀 Features

- 📂 Upload and process documents (PDF/TXT)
- 🔍 Semantic search using vector embeddings
- 🧠 Context-aware answers using LLMs
- ⚡ Fast retrieval with FAISS
- 🧩 Modular and easy-to-extend codebase

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Vector Database:** FAISS  
- **Embeddings:** Sentence Transformers / OpenAI Embeddings  
- **LLM:** OpenAI GPT or equivalent  
- **Framework / UI:** Streamlit / Python scripts  
- **Version Control:** Git & GitHub  

---

## 📁 Project Structure

<img width="804" height="533" alt="image" src="https://github.com/user-attachments/assets/b27103ed-d3e1-4cca-bb94-e585cb24e1a8" />

---

## ⚙️ How It Works (RAG Pipeline)

1. **Document Loading**  
   Documents are loaded and split into smaller chunks.

2. **Embedding Generation**  
   Each chunk is converted into vector embeddings.

3. **Vector Storage**  
   Embeddings are stored in a FAISS vector database.

4. **User Query**  
   User submits a natural language question.

5. **Retrieval**  
   Relevant document chunks are retrieved using semantic similarity.

6. **Answer Generation**  
   Retrieved context is passed to an LLM to generate a final answer.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Raja-Rajeswari-Javvadi/Smart-Document-Question-Answering-System-using-RAG.git
cd Smart-Document-Question-Answering-System-using-RAG
<img width="668" height="637" alt="image" src="https://github.com/user-attachments/assets/9775abe4-edb5-4f6e-aea2-0bc627fdf40c" />

