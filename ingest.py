import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Get API key from environment and strip whitespace
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables. Please set it in your .env file or export it.")

# Strip any whitespace that might cause issues
api_key = api_key.strip()

print(f"DEBUG: Key loaded? {api_key is not None}")
if api_key:
    print(f"DEBUG: Key starts with: {api_key[:7]}...")
    print(f"DEBUG: Key length: {len(api_key)} characters")

def build_vector_db():
    print("🚀 Starting ingestion...")
    
    # --- UPDATED: Look in the 'docs' folder ---
    loader = DirectoryLoader('./docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("❌ No PDF files found in the 'docs' folder!")
        return

    print(f"📄 Loaded {len(documents)} pages from PDF(s).")
    
    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} text chunks.")
    
    # 3. Create Embeddings & Store in ChromaDB
    print("💾 Creating embeddings and saving to database...")
    # Set API key in environment (OpenAIEmbeddings reads from OPENAI_API_KEY env var)
    os.environ["OPENAI_API_KEY"] = api_key
    # Use OpenAI embeddings - text-embedding-3-small is the latest, cost-effective model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    
    # Test the embeddings with a small sample to verify API key works
    print("🔍 Testing API key with a sample embedding...")
    try:
        test_embedding = embeddings.embed_query("test")
        print(f"✅ API key verified! Embedding dimension: {len(test_embedding)}")
    except Exception as e:
        print(f"❌ API key validation failed: {e}")
        print("💡 Please verify your API key is valid and has access to the OpenAI Embeddings API.")
        raise
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./cdf_db"
    )
    print("✅ Success! Database created in './cdf_db' folder.")

if __name__ == "__main__":
    build_vector_db()