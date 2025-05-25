import os
import streamlit as st
from git import Repo
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Constants
REPO_DIR = "cloned_repo"

def clone_repo(repo_url, target_dir=REPO_DIR):
    if os.path.exists(target_dir):
        try:
            os.rename(target_dir, target_dir + "_backup")
            st.info(f"Existing folder renamed to {target_dir + '_backup'}")
        except PermissionError:
            st.error(f"Cannot rename existing folder {target_dir}. Please close apps or delete the folder manually.")
            return None
    try:
        Repo.clone_from(repo_url, target_dir)
        st.success(f"Repository cloned to {target_dir}")
        return target_dir
    except Exception as e:
        st.error(f"Error cloning repo: {e}")
        return None

st.title("💻 GitHub Repo Source Code Q&A with Langchain and OpenAI")

# User inputs OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

repo_url = st.text_input("Enter GitHub repository HTTPS URL (e.g. https://github.com/user/repo.git)")

if st.button("Clone and Index Repo"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key!")
    elif repo_url.strip() == "":
        st.error("Please enter a valid GitHub repository URL.")
    else:
        repo_path = clone_repo(repo_url)
        if repo_path:
            loader = DirectoryLoader(repo_path, glob="**/*.py")
            documents = loader.load()
            st.info(f"Loaded {len(documents)} Python files from repo.")

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(documents)
            st.info(f"Split documents into {len(docs)} chunks.")

            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.from_documents(docs, embeddings)
            st.success("Created FAISS vectorstore from repo documents.")

            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

            st.session_state["qa_chain"] = qa
            st.session_state["vectorstore"] = vectorstore
            st.session_state["ready"] = True

if st.session_state.get("ready", False):
    st.markdown("---")
    query = st.text_input("Ask a question about the code in the repository:")
    if query:
        with st.spinner("Generating answer..."):
            answer = st.session_state["qa_chain"].run(query)
        st.markdown("**Answer:**")
        st.write(answer)

    if st.button("Clear Session"):
        for key in ["qa_chain", "vectorstore", "ready"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()
