import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 1. Configuration - Use exact paths
DATA_PATH = "data/s71200_manual.pdf"
DB_PATH = "industrial_db"

def run_ingestion():
    """
    Main ingestion function that processes the PDF and saves to the vector store.
    This replaces the 'Self-Healing' logic in your RAG app.
    """
    
    # Ensure data exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing PDF at {DATA_PATH}. Please upload it to GitHub.")

    print(f"--- Step 1: Loading PDF using PDFMiner (Stabilized for Industrial Manuals) ---")
    # Using mode="page" ensures page numbers are preserved in metadata
    loader = PDFMinerLoader(DATA_PATH, mode="page")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    print(f"--- Step 2: Splitting Text into Contextual Chunks ---")
    # Industrial manuals benefit from larger chunks to keep technical specs together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print(f"--- Step 3: Generating Embeddings and Saving to ChromaDB ---")
    # Ensure you use the same model as in app.py
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    # Create the vector store. On Streamlit Cloud, persistence is handled automatically
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print(f"--- Success: Vector database saved at {DB_PATH} ---")

# This allows you to test the script locally by running: python ingest_manual.py
if __name__ == "__main__":
    run_ingestion()