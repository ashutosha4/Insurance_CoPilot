import os
import streamlit as st
import pandas as pd
from datetime import datetime
import pickle

from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Constants
FAISS_INDEX_PATH = "faiss_index"
LOG_FILE = "query_logs.csv"

# Load Hugging Face embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load saved FAISS index (if available)
@st.cache_resource
def init_vector_store():
    if os.path.exists(f"{FAISS_INDEX_PATH}.pkl"):
        with open(f"{FAISS_INDEX_PATH}.pkl", "rb") as f:
            return pickle.load(f)
    return None  # Return None if not created yet

# Save FAISS index to disk
def save_vector_store(vector_store):
    with open(f"{FAISS_INDEX_PATH}.pkl", "wb") as f:
        pickle.dump(vector_store, f)

# Ingest uploaded PDF and build vector store
def build_vector_store_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = load_embeddings()
    faiss_store = FAISS.from_documents(chunks, embedding=embeddings)
    save_vector_store(faiss_store)
    return faiss_store

# Load Hugging Face LLM
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=512)

# Generate answer using FAISS + context
def generate_answer(query, vector_store, llm):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    result = llm(prompt)[0]["generated_text"]
    return result.strip(), context

# Log interaction to CSV
def log_interaction(query, context, output):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_context": context,
        "output": output
    }
    df = pd.DataFrame([log_entry])
    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

# Streamlit UI
def main():
    st.set_page_config(page_title="Insurance RAG Co-Pilot", layout="wide")
    st.title("🛡️ Insurance Co-Pilot (RAG-based using FAISS)")
    st.markdown("Upload policy documents, SOPs, or training material and ask domain-specific questions. All responses are generated using context retrieved from your uploaded documents.")

    vector_store = init_vector_store()
    llm = load_llm()

    # Upload document
    uploaded_file = st.file_uploader("📄 Upload a PDF (Policy, Claim Guide, SOP)", type=["pdf"])
    if uploaded_file is not None:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        vector_store = build_vector_store_from_pdf("temp_uploaded.pdf")
        st.success("✅ Document uploaded and indexed successfully!")

    # Ask question if vector store exists
    if vector_store is not None:
        query = st.text_input("🔍 Ask your insurance-related question:")
        if query:
            with st.spinner("Generating answer..."):
                output, context = generate_answer(query, vector_store, llm)
                log_interaction(query, context, output)

            st.subheader("🧠 Answer:")
            st.write(output)

            st.subheader("📚 Retrieved Context:")
            with st.expander("Show Context"):
                st.write(context)

            st.success("✅ Logged for hallucination tracking.")
    else:
        st.info("ℹ️ Please upload a document to enable question answering.")

    # Show recent logs
    if os.path.exists(LOG_FILE):
        with st.expander("🗂️ View Recent Query Logs"):
            df = pd.read_csv(LOG_FILE)
            st.dataframe(df.tail(10), use_container_width=True)

if __name__ == "__main__":
    main()
