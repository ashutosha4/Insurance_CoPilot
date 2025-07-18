import os
import streamlit as st
import pandas as pd
from datetime import datetime

from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS
from transformers import pipeline
import pickle

# Constants
FAISS_INDEX_PATH = "faiss_index"
LOG_FILE = "query_logs.csv"

# Load embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or initialize FAISS vector store
@st.cache_resource
def init_vector_store():
    if os.path.exists(f"{FAISS_INDEX_PATH}.pkl"):
        with open(f"{FAISS_INDEX_PATH}.pkl", "rb") as f:
