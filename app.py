import os
import streamlit as st
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import tempfile
import pickle

# Constants
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"
FAISS_INDEX_PATH = "faiss_index"

# Function to load embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# Function to load lightweight Flan-T5 model
@st.cache_resource
def load_llm():
    hf_pipeline = pipeline("text2text-generation", model=LLM_MODEL_NAME, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=hf_pipeline)

# Function to load or initialize FAISS vector store
@st.cache_resource
def init_vector_store():
    if os.path.exists(f"{FAISS_INDEX_PATH}.pkl") and os.path.exists(f"{FAISS_INDEX_PATH}.index"):
        with open(f"{FAISS_INDEX_PATH}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return FAISS.from_texts([], embedding=load_embeddings())

# Function to save FAISS index
def save_vector_store(vector_store):
    with open(f"{FAISS_INDEX_PATH}.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    vector_store.save_local(FAISS_INDEX_PATH)

# Function to process uploaded documents
def process_documents(uploaded_files, vector_store):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    new_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load_and_split(text_splitter)
        new_docs.extend(docs)

        os.remove(tmp_path)

    if new_docs:
        vector_store.add_documents(new_docs)
        save_vector_store(vector_store)
        st.success("Documents uploaded and indexed successfully!")

# Function to run the RAG chain
def run_query(llm, vector_store, query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)

# Streamlit UI
def main():
    st.set_page_config(page_title="Insurance CoPilot (RAG)", layout="wide")
    st.title("🧠 Insurance CoPilot: RAG over Policy Docs")
    st.markdown("Ask questions about uploaded policy documents, SOPs, or claims guides.")

    vector_store = init_vector_store()
    llm = load_llm()

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Index Uploaded PDFs"):
        if uploaded_files:
            process_documents(uploaded_files, vector_store)
        else:
            st.warning("Please upload at least one PDF.")

    # Ask a question
    st.subheader("Ask a question")
    user_query = st.text_input("Enter your question here:")
    if user_query:
        with st.spinner("Generating answer..."):
            answer = run_query(llm, vector_store, user_query)
            st.markdown(f"### Answer:\n{answer}")

if __name__ == "__main__":
    main()
