import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment & Setup
load_dotenv()
st.set_page_config(page_title="CDF Knowledge Assistant", page_icon="🛡️", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px; margin-bottom: 10px;}
    .stSpinner {text-align: center;}
</style>
""", unsafe_allow_html=True)

# 2. Sidebar Controls
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Settings")
    st.markdown("---")
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.0, help="0 is strict, 1 is creative.")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("Powered by CDF RAG V1")

# 3. Initialize Retrieval Engine (Cached for speed)
@st.cache_resource
def get_retriever():
    """Initialize and return the vector store retriever."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory="./cdf_db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 4})

# Initialize retriever
try:
    retriever = get_retriever()
except Exception as e:
    st.error(f"🚨 Error loading database: {str(e)}")
    st.info("💡 Make sure you have run 'python3 ingest.py' first to create the vector database.")
    st.stop()

# 4. Main Chat Interface
st.header("🛡️ CDF Knowledge Assistant")
st.caption("Ask specific questions about the uploaded PDF manuals.")

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to answer questions based on the CDF documents."}]

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Handle User Input
if prompt := st.chat_input("Ex: What are the volunteer responsibilities?"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching CDF Knowledge Base..."):
            try:
                # Retrieve relevant documents
                docs = retriever.invoke(prompt)
                
                # Format context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Create prompt template
                prompt_template = """You are a specialized technical assistant for the Community Dreams Foundation (CDF).

RULES:
1. Use ONLY the provided context below to answer the question.
2. If the answer is not in the context, explicitly say: "I cannot find this information in the CDF documentation."
3. Do not make up answers.
4. Keep answers clear, professional, and structured (use bullet points if needed).

CONTEXT:
{context}

QUESTION: 
{question}

ANSWER:"""
                
                # Create the chain
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
                
                # Format the prompt
                formatted_prompt = prompt_template.format(context=context, question=prompt)
                
                # Get response from LLM
                response = llm.invoke(formatted_prompt)
                result_text = response.content

                # Format Sources properly
                unique_sources = set()
                for doc in docs:
                    # Get filename, default to 'Unknown' if missing
                    source_name = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                    # Get page number if available
                    page_num = doc.metadata.get('page', '')
                    ref = f"{source_name} (Page {page_num})" if page_num else source_name
                    unique_sources.add(ref)

                # Final Display
                st.markdown(result_text)
                
                # Expandable Source Section
                if unique_sources:
                    with st.expander("📚 View Sources"):
                        for src in unique_sources:
                            st.markdown(f"- `{src}`")

                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": result_text})
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"An error occurred: {error_msg}")
                st.exception(e)
