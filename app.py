import streamlit as st
import tempfile
import os

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Document Chatbot")
st.write("Upload a PDF and ask questions using Retrieval-Augmented Generation (RAG).")


# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return embedding_model, tokenizer, llm


embedding_model, tokenizer, llm = load_models()


# -----------------------------
# Extract PDF Text
# -----------------------------
def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Create Text Chunks
# -----------------------------
def create_chunks(text, chunk_size=500, overlap=100):

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# Create FAISS Database
# -----------------------------
def create_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# -----------------------------
# Retrieve Relevant Chunks
# -----------------------------
def retrieve_documents(question, chunks, index, top_k=3):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    distances, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for i in indices[0]:

        if 0 <= i < len(chunks):

            results.append(chunks[i])

    return results


# -----------------------------
# Generate Answer
# -----------------------------
def generate_answer(question, context):

    prompt = f"""
Answer the question using only the information provided in the context.

If the answer is not available in the context, say:
"I could not find the answer in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = llm.generate(
            **inputs,
            max_new_tokens=120
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# -----------------------------
# PDF Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    with st.spinner("Processing your document..."):

        text = extract_text_from_pdf(uploaded_file)

        if not text.strip():

            st.error("Could not extract text from this PDF.")

            st.stop()

        chunks = create_chunks(text)

        index = create_vector_database(chunks)

    st.success(
        f"✅ Document processed successfully! Created {len(chunks)} text chunks."
    )


    # -----------------------------
    # Question Input
    # -----------------------------

    question = st.text_input(
        "💬 Ask a question about your document:"
    )


    if question:

        with st.spinner("Searching document and generating answer..."):

            retrieved_chunks = retrieve_documents(
                question,
                chunks,
                index,
                top_k=3
            )

            context = "\n\n".join(retrieved_chunks)

            answer = generate_answer(
                question,
                context
            )


        # -----------------------------
        # Display Answer
        # -----------------------------

        st.subheader("🤖 AI Answer")

        st.write(answer)


        # -----------------------------
        # Display Sources
        # -----------------------------

        with st.expander("📚 Retrieved Context"):

            for i, chunk in enumerate(retrieved_chunks):

                st.markdown(
                    f"**Source {i + 1}**"
                )

                st.write(chunk)

                st.divider()

else:

    st.info(
        "👆 Upload a PDF document to start chatting with your document."
  )
