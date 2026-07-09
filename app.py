import streamlit as st
import os
import time

from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Samsung AI Manual Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.stApp{
background:#0d1117;
}

.hero{
padding:35px;
border-radius:22px;
background:linear-gradient(135deg,#1e3c72,#2a5298,#6a11cb);
text-align:center;
margin-bottom:25px;
box-shadow:0 12px 30px rgba(0,0,0,.35);
}

.hero h1{
color:white;
font-size:54px;
font-weight:800;
margin-bottom:8px;
}

.hero p{
color:#ECECEC;
font-size:20px;
}

.card{

background:#161b22;

padding:22px;

border-radius:18px;

border:1px solid #30363d;

margin-bottom:20px;

}

.metric-card{

background:#202938;

padding:18px;

border-radius:15px;

text-align:center;

}

.stButton>button{

width:100%;

padding:12px;

border-radius:12px;

background:linear-gradient(90deg,#00c6ff,#0072ff);

color:white;

border:none;

font-weight:bold;

font-size:16px;

}

.stButton>button:hover{

transform:scale(1.02);

transition:.2s;

}

hr{
border:1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HERO
# ---------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🤖 Samsung AI Manual Assistant</h1>

<p>
Retrieval-Augmented Generation (RAG)
powered by OpenAI & LangChain
</p>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

defaults = {
    "messages": [],
    "vectorstore": None,
    "retriever": None,
    "chain": None,
    "ready": False,
    "chunks": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.title("⚙️ Control Panel")

    api_key = st.secrets.get("OPENAI_API_KEY", "")

    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password"
        )

    st.divider()

    st.subheader("📊 Models")

    st.success("GPT-4o-mini")

    st.success("text-embedding-3-small")

    st.success("ChromaDB")

    st.divider()

    st.subheader("📈 Statistics")

    docs_placeholder = st.empty()
    chunk_placeholder = st.empty()
    msg_placeholder = st.empty()

    docs_placeholder.metric(
        "Documents",
        1
    )

    chunk_placeholder.metric(
        "Chunks",
        st.session_state.chunks
    )

    msg_placeholder.metric(
        "Messages",
        len(st.session_state.messages)
    )

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------
# MAIN PANEL
# ---------------------------------------------------

st.markdown("""
<div class="card">

## 📖 Knowledge Base

This application automatically loads the Samsung Washing
Machine Manual from the project directory.

No upload is required.

</div>
""", unsafe_allow_html=True)

st.info(
    "Place **samsung_manual.html** in the same folder as **app.py**."
)
# ==========================================================
# SECTION 2 : BUILD RAG KNOWLEDGE BASE
# ==========================================================

MANUAL_PATH = "samsung_manual.html"


@st.cache_resource(show_spinner=False)
def initialize_rag(api_key):
    """
    Loads the Samsung HTML manual and creates the
    vector database only once.
    """

    if not os.path.exists(MANUAL_PATH):
        raise FileNotFoundError(
            f"{MANUAL_PATH} not found."
        )

    # -----------------------------
    # Load HTML
    # -----------------------------
    loader = BSHTMLLoader(MANUAL_PATH)

    documents = loader.load()

    # -----------------------------
    # Split into chunks
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # -----------------------------
    # Embeddings
    # -----------------------------
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )

    # -----------------------------
    # Vector Database
    # -----------------------------
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":4}
    )

    # -----------------------------
    # LLM
    # -----------------------------
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert Samsung Washing Machine assistant.

Answer ONLY using the provided context.

If the answer is not available in the manual,
say:

"I couldn't find that information in the manual."

Keep your answers concise and accurate.

Question:
{question}

Context:
{context}

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return (
        vectorstore,
        retriever,
        rag_chain,
        len(chunks)
    )


# ==========================================================
# INITIALIZE
# ==========================================================

if not st.session_state.ready:

    if not api_key:

        st.warning("Please enter your OpenAI API Key.")

        st.stop()

    with st.spinner("📚 Loading Samsung Manual..."):

        try:

            (
                vectorstore,
                retriever,
                chain,
                total_chunks
            ) = initialize_rag(api_key)

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.chain = chain
            st.session_state.ready = True
            st.session_state.chunks = total_chunks

        except Exception as e:

            st.error(e)

            st.stop()

# ==========================================================
# UPDATE DASHBOARD
# ==========================================================

docs_placeholder.metric(
    "Documents",
    1
)

chunk_placeholder.metric(
    "Chunks",
    st.session_state.chunks
)

msg_placeholder.metric(
    "Messages",
    len(st.session_state.messages)
)

st.success(
    "✅ Samsung Manual Loaded Successfully"
)
# ==========================================================
# SECTION 3 : CHAT INTERFACE
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class="card">

<h2>💬 Ask the AI Assistant</h2>

Ask any question related to the Samsung Washing Machine Manual.

</div>
""",
unsafe_allow_html=True
)

# ----------------------------
# Display Previous Messages
# ----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# Chat Input
# ----------------------------

question = st.chat_input(
    "Ask something about your washing machine..."
)

if question:

    # ----------------------------
    # Display User Message
    # ----------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # ----------------------------
    # Assistant Response
    # ----------------------------

    with st.chat_message("assistant"):

        thinking = st.empty()

        thinking.info(
            "🔍 Searching the manual..."
        )

        try:

            # Retrieve relevant chunks
            retrieved_docs = (
                st.session_state.retriever.invoke(question)
            )

            # Generate response
            answer = st.session_state.chain.invoke(question)

            if hasattr(answer, "content"):
                response = answer.content
            else:
                response = str(answer)

            thinking.empty()

            placeholder = st.empty()

            streamed_text = ""

            # Typing animation
            for word in response.split():

                streamed_text += word + " "

                placeholder.markdown(
                    streamed_text + "▌"
                )

                time.sleep(0.02)

            placeholder.markdown(streamed_text)

            # Save assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            # ----------------------------
            # Retrieved Context
            # ----------------------------

            with st.expander(
                "📚 Retrieved Context",
                expanded=False
            ):

                for i, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):

                    st.markdown(
                        f"### Chunk {i}"
                    )

                    st.write(doc.page_content)

                    st.divider()

            # ----------------------------
            # Response Statistics
            # ----------------------------

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Retrieved Chunks",
                len(retrieved_docs)
            )

            col2.metric(
                "Messages",
                len(st.session_state.messages)
            )

            col3.metric(
                "Model",
                "GPT-4o-mini"
            )

        except Exception as e:

            thinking.empty()

            st.error(e)

# ----------------------------
# Update Sidebar Metrics
# ----------------------------

msg_placeholder.metric(
    "Messages",
    len(st.session_state.messages)
)
# ==========================================================
# SECTION 4 : FINAL UI POLISH
# ==========================================================

st.markdown("---")

# ----------------------------
# Suggested Questions
# ----------------------------

st.markdown("## 💡 Suggested Questions")

col1, col2 = st.columns(2)

with col1:

    if st.button("🧺 What is Daily Wash?"):
        st.session_state["suggested_question"] = "What is Daily Wash?"

    if st.button("⚠️ Show all warning messages"):
        st.session_state["suggested_question"] = "What are the warning messages in the manual?"

with col2:

    if st.button("🌡 Which wash mode is suitable for delicate clothes?"):
        st.session_state["suggested_question"] = "Which wash mode is suitable for delicate clothes?"

    if st.button("🧼 How do I clean the washing machine?"):
        st.session_state["suggested_question"] = "How do I clean the washing machine?"


# ----------------------------
# Download Conversation
# ----------------------------

if st.session_state.messages:

    history = ""

    for message in st.session_state.messages:

        history += (
            f"{message['role'].upper()}:\n"
            f"{message['content']}\n\n"
        )

    st.download_button(
        label="📥 Download Conversation",
        data=history,
        file_name="chat_history.txt",
        mime="text/plain"
    )


# ----------------------------
# Knowledge Base Summary
# ----------------------------

st.markdown("## 📊 Knowledge Base")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        f"""
### Documents

1 HTML Manual
"""
    )

with col2:

    st.success(
        f"""
### Chunks

{st.session_state.chunks}
"""
    )

with col3:

    st.warning(
        """
### LLM

GPT-4o-mini
"""
    )


# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.markdown(
"""
<div style='
text-align:center;
padding:25px;
border-radius:18px;
background:linear-gradient(90deg,#1e3c72,#2a5298,#6a11cb);
color:white;
margin-top:30px;
'>

<h2>🤖 Samsung AI Manual Assistant</h2>

<p>
Powered by Streamlit • LangChain • OpenAI • ChromaDB
</p>

<p>
Built with ❤️ using Retrieval-Augmented Generation (RAG)
</p>

</div>
""",
unsafe_allow_html=True
)
