import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Paths
DOCUMENTS_PATH = "documents"
FAISS_INDEX_PATH = "faiss_index"
LOG_FILE = "query_logs.txt"

# Load LLM
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)

# Load Prompt
@st.cache_resource
def load_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an insurance expert. Use the following context to answer the question.
        If you don't know the answer, say "I don't know."

        Context: {context}
        Question: {question}
        Answer:
        """
    )

# Save uploaded documents
def save_uploaded_file(uploaded_file):
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
    file_path = os.path.join(DOCUMENTS_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# Index documents using FAISS
def build_vector_store(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([documents])
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store

# Load FAISS index
def load_vector_store():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Handle query
def run_query(llm, vector_store, query):
    prompt = load_prompt()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    result = chain.run(query)
    # Log interaction
    with open(LOG_FILE, "a") as log:
        log.write(f"Query: {query}\n")
        context = vector_store.similarity_search(query, k=2)
        context_text = "\n".join([doc.page_content for doc in context])
        log.write(f"Retrieved Context:\n{context_text}\n")
        log.write(f"Response: {result}\n{'-'*80}\n")
    return result

# Streamlit UI
def main():
    st.set_page_config(page_title="Insurance Co-Pilot", layout="wide")
    st.title("🧠 Insurance Co-Pilot (RAG-based)")
    st.write("Supports policy admin, claims, customer support, and training.")

    llm = load_llm()

    # Upload PDFs
    uploaded_file = st.file_uploader("Upload a policy or claims PDF", type="pdf")
    if uploaded_file:
        path = save_uploaded_file(uploaded_file)
        with st.spinner("Indexing document..."):
            full_text = extract_text_from_pdf(path)
            vector_store = build_vector_store(full_text)
        st.success("Document indexed and ready!")
    else:
        # Try to load existing index
        try:
            vector_store = load_vector_store()
        except Exception as e:
            st.warning("No index found. Please upload a document first.")
            return

    # Query input
    user_query = st.text_input("Ask a question (e.g., What documents are needed for claims adjudication?)")
    if st.button("Get Answer") and user_query:
        with st.spinner("Thinking..."):
            answer = run_query(llm, vector_store, user_query)
        st.markdown(f"### 💬 Answer:\n{answer}")

if __name__ == "__main__":
    main()
