import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline

# Constants
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"

# File upload
st.set_page_config(page_title="Insurance RAG Copilot", layout="wide")
st.title("🧠 Insurance RAG Copilot")
uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

# Load Embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# Initialize FAISS vector store
@st.cache_resource
def init_vector_store():
    if os.path.exists(f"{FAISS_INDEX_PATH}.pkl") and os.path.exists(f"{FAISS_INDEX_PATH}.index"):
        with open(f"{FAISS_INDEX_PATH}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return None  # No documents indexed yet

# Save vector store
def save_vector_store(vector_store):
    with open(f"{FAISS_INDEX_PATH}.pkl", "wb") as f:
        pickle.dump(vector_store, f)

# Load FLAN-T5 model
@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=512)

# Process PDF files
def process_documents(files):
    texts = []
    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                texts.append(content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text("\n".join(texts))

# Index documents
def index_documents(texts, embeddings):
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    save_vector_store(vector_store)
    return vector_store

# RAG QA Chain
def run_query(llm, vector_store, query):
    prompt_template = """
    You are a helpful assistant for insurance documents.
    Use the following context to answer the question concisely:
    
    Context:
    {context}
    
    Question:
    {question}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    result = chain.run(query)
    return result

# Main app logic
def main():
    embeddings = load_embeddings()
    vector_store = init_vector_store()
    llm = load_llm()

    # Document indexing
    if uploaded_files and st.button("Index Uploaded PDFs"):
        with st.spinner("Processing and indexing documents..."):
            texts = process_documents(uploaded_files)
            if texts:
                vector_store = index_documents(texts, embeddings)
                st.success("Documents indexed successfully.")
            else:
                st.warning("No extractable text found in uploaded PDFs.")

    # Query input
    user_query = st.text_input("Ask a question about the documents")

    # Query processing
    if user_query:
        if vector_store is None:
            st.error("Please upload and index documents before asking questions.")
        else:
            with st.spinner("Generating answer..."):
                answer = run_query(llm, vector_store, user_query)
                st.markdown(f"### Answer:\n{answer}")

if __name__ == "__main__":
    main()
