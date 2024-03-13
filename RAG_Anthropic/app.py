from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import streamlit as st
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

st.set_page_config(layout = "wide")
DOCS_DIR = os.path.abspath("./uploaded_docs")


anthropic = Anthropic(api_key=API_KEY)

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

