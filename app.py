try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If on Windows, pysqlite3 won't exist. We just skip this step.
    pass

import streamlit as st
import os
from ingest_manual import run_ingestion 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

# 1. Page Configuration
st.set_page_config(page_title="Industrial AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Siemens S7-1200 Expert")
st.markdown("Ask technical questions based on the 1,597-page manual.")

# 2. Setup Models & Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets in Advanced Settings.")
    st.stop()

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 3. Self-Healing Database Logic
DB_PATH = "industrial_db"

@st.cache_resource(show_spinner="Initializing Manual Database... This may take a minute on first run.")
def load_vector_db():
    # If the folder doesn't exist, run the ingestion script automatically
    if not os.path.exists(DB_PATH):
        try:
            run_ingestion()
        except Exception as e:
            st.error(f"Failed to initialize database: {e}")
            st.stop()
    
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

vector_db = load_vector_db()

# 4. Create the Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# 5. Chat Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input Logic
if prompt := st.chat_input("Ex: What is the maximum number of local I/O modules?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing manual..."):
            response = qa_chain.invoke(prompt)
            answer = response["result"]
            sources = response["source_documents"]

            st.markdown(answer)
            
            with st.expander("📚 View Manual Sources"):
                for i, doc in enumerate(sources):
                    page_num = doc.metadata.get("page", "Unknown")
                    st.markdown(f"**Source {i+1}:** Page {page_num + 1}")
                    st.caption(f"Context: {doc.page_content[:200]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})