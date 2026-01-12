import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Load secret key
load_dotenv()

# 2. UI Configuration
st.set_page_config(page_title="CDF Onboarding Bot", page_icon="🤝")
st.markdown("<h1 style='color: #007BFF;'>🤝 CDF Volunteer Onboarding Bot</h1>", unsafe_allow_html=True)
st.write("Welcome! I'm here to help new volunteers navigate the **Community of Developers (CDF)**.")

# 3. Setup OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY in your .env file!")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 4. Load Knowledge Base
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # 4a. Define Prompt Template
    template = """You are a CDF Document Expert. Use the provided context to answer the user's question accurately.
    
    - If the user asks for a list, provide all relevant items found in the context.
    - 'How do I join CDF?' is answered in the context; ensure you look for 'application' or 'cdreamstream.org'.
    - If the answer is truly missing, say you don't know.
    - Always cite your answers professionally.

    Context: {context}

    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # 5. Build the Chain with MMR Search
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # search_type="mmr" helps find diverse info across different PDF pages
        retriever=vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 5, "fetch_k": 20}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
else:
    st.warning("⚠️ Knowledge base not found. Please run 'python build_index.py' first.")
    st.stop()

# 6. Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about CDF volunteering..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching CDF guides..."):
            result = qa_chain.invoke({"query": prompt})
            response = result["result"]
            sources = result["source_documents"]

            st.markdown(response)

            if sources:
                with st.expander("📚 View Sources"):
                    for doc in sources:
                        source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        page_num = doc.metadata.get('page', 0) + 1
                        st.write(f"- **File:** {source_name} (Page {page_num})")

            st.session_state.messages.append({"role": "assistant", "content": response})