from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the manual
loader = PyPDFLoader("data/manual.pdf")
pages = loader.load()
print(f"Successfully loaded {len(pages)} pages.")

# 2. Split into chunks
# RecursiveCharacterTextSplitter is best for manuals because it 
# respects paragraphs and sentences.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(pages)
print(f"Created {len(chunks)} text chunks for the knowledge base.")

# 3. Preview a chunk to verify context
if chunks:
    print("\n--- Sample Chunk (from Page 1) ---")
    print(chunks[0].page_content[:300])
    print(f"\nMetadata: {chunks[0].metadata}")