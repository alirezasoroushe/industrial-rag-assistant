__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import streamlit as st
import os
from dotenv import load_dotenv

# Modern 2026 Import Path for legacy chains
from langchain_classic.chains import RetrievalQA 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Page Configuration
st.set_page_config(page_title="Industrial AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Siemens S7-1200 Expert")
st.markdown("Ask technical questions based on the 1,597-page manual.")

# 2. Load Environment & Setup Models
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please set the GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 3. Connect to the Vector Memory
# Point to the folder that contains 'chroma.sqlite3'
DB_PATH = "industrial_db" 

if not os.path.exists(os.path.join(DB_PATH, "chroma.sqlite3")):
    st.error(f"Critical Error: 'chroma.sqlite3' not found in {DB_PATH}. Check your GitHub upload.")
    st.stop()

vector_db = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=embeddings
)

# 4. Create the Retrieval Chain with Sources
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True # This enables citations
)

# 5. Chat Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input Logic
if prompt := st.chat_input("Ex: How do I wire a digital input?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing manual..."):
            # The result is now a dictionary containing 'result' and 'source_documents'
            response = qa_chain.invoke(prompt)
            answer = response["result"]
            sources = response["source_documents"]

            st.markdown(answer)
            
            # --- Citation UI Section ---
            with st.expander("📚 View Manual Sources"):
                for i, doc in enumerate(sources):
                    page_num = doc.metadata.get("page", "Unknown")
                    # Increment page by 1 because PDFs start at index 0
                    st.markdown(f"**Source {i+1}:** Page {page_num + 1}")
                    st.caption(f"Context: {doc.page_content[:200]}...")
            
            # Save assistant response to history

            st.session_state.messages.append({"role": "assistant", "content": answer})

