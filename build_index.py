import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Config
DATA_PATH = "data/"
INDEX_PATH = "faiss_index"

def build():
    documents = []
    # 1. Load all PDFs from data/ folder
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"📄 Loading: {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    # 2. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 3. Create searchable embeddings
    print("🔗 Creating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Save to local FAISS index
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_PATH)
    print(f"✅ Success! Search index saved to '{INDEX_PATH}'")

if __name__ == "__main__":
    build()