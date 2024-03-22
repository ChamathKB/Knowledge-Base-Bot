from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS

import streamlit as st
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

st.set_page_config(layout = "wide")
DOCS_DIR = os.path.abspath("./uploaded_docs")


anthropic = Anthropic(api_key=API_KEY)
document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="query")


def chat(user_input):
    stream = anthropic.completions.create(
	model="claude-2.1",
	max_tokens_to_sample=350,
	prompt=f"{HUMAN_PROMPT} {user_input} {AI_PROMPT}",
	stream=True,
	)
    return stream


def document_loader(DOCS_DIR):
  with st.sidebar:
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files = True)
        submitted = st.form_submit_button("Upload!")

    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name),"wb") as f:
                f.write(uploaded_file.read())


with st.sidebar:
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)


vector_store_path = "vectorstore.pkl"


raw_documents = DirectoryLoader(DOCS_DIR).load()


vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    with st.sidebar:
        if raw_documents:
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")
