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

# Constant for the form link
FORM_LINK = "https://docs.google.com/forms/d/e/1FAIpQLSeb1vE7-hXGgtqwI2mrabMB_OkFcOazp7W6oM3RaGgCegJW1w/viewform?usp=dialog"

if os.path.exists("faiss_index"):
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # 4a. Updated Prompt Template for Partial Answers & Guardrails
    template = """You are a CDF Document Expert. Use the provided context to answer the user's question accurately.

    - If the user's question has multiple parts, answer every part you can find in the context.
    - If you find some information but not all (e.g., you find registration steps but not core values), provide the information you found and then add: 
      "Note: I couldn't find information regarding [the missing part] in our documents. For that specific inquiry, please fill out our Support Form."
    - If the entire question is UNRELATED to CDF (personal questions, jokes, or general trivia), politely state that you only assist with CDF documentation and do NOT provide the form link.
    - If the question IS about CDF but you find NOTHING at all, use this specific fallback: 
      "I'm sorry, I couldn't find that information in our current documentation. Please fill out our Support Form and someone from our team will respond to you via email."

Context: {context}

Question: {question}
Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # 5. Build the Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 8, "fetch_k": 30}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
else:
    st.warning("⚠️ Knowledge base not found. Please run 'python build_index.py' first.")
    st.stop()

# 6. Interface & Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.info("Tip: If you have multiple questions, I can answer them all at once if they are in the docs!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "Support Form" in message["content"] and message["role"] == "assistant":
            st.link_button("📋 Open Support Form", FORM_LINK)

if prompt := st.chat_input("Ask about CDF volunteering..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching CDF guides..."):
            result = qa_chain.invoke({"query": prompt})
            response = result["result"]
            sources = result["source_documents"]

            st.markdown(response)
            
            # Show button if the AI mentioned the Support Form (either full fallback or partial)
            if "Support Form" in response:
                st.link_button("📋 Open Support Form", FORM_LINK)

            # Show sources if any were used to generate the answer
            if sources and "only assist with CDF" not in response:
                with st.expander("📚 View Sources"):
                    for doc in sources:
                        source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        page_num = doc.metadata.get('page', 0) + 1
                        st.write(f"- **File:** {source_name} (Page {page_num})")

            st.session_state.messages.append({"role": "assistant", "content": response})