# 🤖 AI Document Chatbot using RAG

An AI-powered document question-answering chatbot built using **Retrieval-Augmented Generation (RAG)**. The application allows users to upload a PDF document and ask questions based on its content.

## 🚀 Live Demo

👉 https://ai-document-chatbot-rag-aupcqip64ahpqexeyuwgef.streamlit.app/

## 📌 Project Overview

This project demonstrates how RAG can connect documents with an AI language model to provide context-based answers.

The system processes the uploaded document, divides the text into smaller chunks, converts the chunks into embeddings, stores them in a FAISS vector database, retrieves relevant information, and generates an answer using an LLM.

📂 Project Structure
AI-Document-Chatbot-RAG/
│
├── app.py
├── RAG_Document_Chatbot.ipynb
├── AI_Notes_for_RAG_Project.pdf
├── requirements.txt
└── README.md

## 🔄 RAG Pipeline

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embeddings
     ↓
FAISS Vector Database
     ↓
Semantic Similarity Search
     ↓
Relevant Context Retrieval
     ↓
FLAN-T5 LLM
     ↓
Final Answer

✨ Features
📄 Upload PDF documents
🔍 Extract text from documents
✂️ Split text into smaller chunks
🧠 Generate text embeddings
🗃️ Store embeddings using FAISS
🔎 Retrieve relevant document content
🤖 Generate context-based answers
🌐 Streamlit web interface