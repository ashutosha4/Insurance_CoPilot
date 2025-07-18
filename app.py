import os
import streamlit as st
import pandas as pd
from datetime import datetime

from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

# Constants
CHROMA_DIR = "chroma_db"
LOG_FILE = "query_logs.csv"

# Initialize embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector store
@st.cache_resource
def init_vector_store():
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=load_embeddings())

# Initialize Hugging Face LLM
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=512)

# Ingest uploaded PDF into ChromaDB
def ingest_document(file_path, vector_store):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store.add_documents(chunks)

# Run RAG pipeline: retrieve relevant context + generate answer
def generate_answer(query, vector_store, llm):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    result = llm(prompt)[0]["generated_text"]
    return result.strip(), context

# Log query, context, and output to CSV
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
    st.title("🛡️ Insurance Co-Pilot (RAG-based)")
    st.markdown("Use this tool to ask questions related to policy administration, claims adjudication, training, and more.")

    vector_store = init_vector_store()
    llm = load_llm()

    # Upload and ingest PDF
    uploaded_file = st.file_uploader("📄 Upload a PDF document (policy, SOP, rules, etc.)", type=["pdf"])
    if uploaded_file is not None:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        ingest_document("temp_uploaded.pdf", vector_store)
        st.success("✅ Document uploaded and processed.")

    # Query input
    query = st.text_input("🔍 Ask your insurance-related question:")
    if query:
        with st.spinner("Generating answer..."):
            output, context = generate_answer(query, vector_store, llm)
            log_interaction(query, context, output)

        st.subheader("🧠 Answer:")
        st.write(output)

        st.subheader("📚 Retrieved Context:")
        with st.expander("View Context"):
            st.write(context)

        st.success("✅ Interaction logged for hallucination tracking.")

    # Display logs (optional)
    if os.path.exists(LOG_FILE):
        with st.expander("🗂️ View Logged Interactions"):
            df = pd.read_csv(LOG_FILE)
            st.dataframe(df.tail(10), use_container_width=True)

if __name__ == "__main__":
    main()
