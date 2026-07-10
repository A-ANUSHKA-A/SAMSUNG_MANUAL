from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
import time

from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
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

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}


/* APP BACKGROUND */

.stApp{

background:
radial-gradient(circle at top left,#172554,#020617 40%);

color:white;

}


/* HERO */

.hero{

padding:45px;

border-radius:30px;

background:
linear-gradient(
135deg,
rgba(30,64,175,.9),
rgba(124,58,237,.9),
rgba(14,165,233,.9)
);

text-align:center;

box-shadow:
0 20px 50px rgba(0,0,0,.5);

animation:
fadeIn 1s ease-in-out;

}


.hero h1{

font-size:60px;

font-weight:900;

letter-spacing:-2px;

background:
linear-gradient(
90deg,
#ffffff,
#bae6fd
);

-webkit-background-clip:text;

color:transparent;

}


.hero p{

font-size:22px;

color:#e0f2fe;

}



/* CARDS */


.card{

background:
rgba(255,255,255,.08);

backdrop-filter:
blur(15px);

padding:25px;

border-radius:25px;

border:

1px solid rgba(255,255,255,.15);

box-shadow:

0 15px 40px rgba(0,0,0,.3);

margin-bottom:25px;

}



/* CHAT AREA */


[data-testid="stChatMessage"]{

background:

rgba(255,255,255,.06);

border-radius:20px;

padding:10px;

margin-bottom:15px;

}



/* INPUT */


[data-testid="stChatInput"]{

border-radius:20px;

background:#111827;

}



/* BUTTONS */


.stButton button{


background:

linear-gradient(
90deg,
#06b6d4,
#2563eb
);


border:none;

border-radius:15px;

height:50px;

font-weight:700;

font-size:16px;

color:white;


transition:.3s;

}


.stButton button:hover{


transform:
translateY(-3px)
scale(1.03);


box-shadow:

0 10px 25px rgba(37,99,235,.5);


}



/* METRIC CARDS */


[data-testid="metric-container"]{


background:

rgba(255,255,255,.08);


padding:20px;

border-radius:20px;


border:

1px solid rgba(255,255,255,.15);


}



/* SIDEBAR */


section[data-testid="stSidebar"]{


background:

linear-gradient(
180deg,
#020617,
#111827
);


}



/* EXPANDER */


.streamlit-expanderHeader{


background:

rgba(255,255,255,.08);


border-radius:15px;


}



/* ANIMATION */


@keyframes fadeIn{


from{

opacity:0;

transform:
translateY(20px);

}


to{

opacity:1;

transform:
translateY(0);

}

}


</style>
""",
unsafe_allow_html=True)
# ---------------------------------------------------
# HERO
# ---------------------------------------------------

st.markdown("""
<div class="hero">

<h1>
🤖 Samsung AI Copilot
</h1>

<p>
Your intelligent washing machine companion powered by RAG,
LangChain & Gemini
</p>

<div style="
margin-top:20px;
font-size:18px;
">

🟢 Manual Loaded  
&nbsp;&nbsp;•&nbsp;&nbsp;
🔎 Semantic Search Active  
&nbsp;&nbsp;•&nbsp;&nbsp;
⚡ AI Ready

</div>

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

    api_key = st.secrets.get("GOOGLE_API_KEY", "")


    st.divider()

    st.subheader("📊 Models")

    st.success("Gemini 2.5 Flash")

    st.success("Gemini Embedding-001")

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
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
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
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        google_api_key=api_key
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


    # -----------------------------
    # Format Retrieved Documents
    # -----------------------------
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )


    # -----------------------------
    # RAG Chain
    # -----------------------------
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


    # -----------------------------
    # Return Components
    # -----------------------------
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

        st.warning("Gemini API Key is not configured. Add GOOGLE_API_KEY in Streamlit Secrets.")

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
st.markdown("""
<div class="card">

<h3>
🧠 AI System Status
</h3>


<table style="width:100%">

<tr>
<td>📚 Knowledge Base</td>
<td>✅ Samsung Manual Indexed</td>
</tr>


<tr>
<td>🔍 Retrieval Engine</td>
<td>✅ Chroma Vector Search</td>
</tr>


<tr>
<td>🤖 Language Model</td>
<td>✅ Gemini 2.5 Flash</td>
</tr>


<tr>
<td>⚡ Response Mode</td>
<td>Streaming Enabled</td>
</tr>


</table>

</div>

""",
unsafe_allow_html=True)

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

        thinking.markdown(
"""
<div class="card">

<h3>
🤖 Thinking...
</h3>

<p>
🔎 Searching Samsung manual<br>
🧩 Retrieving relevant knowledge<br>
✨ Generating answer

</p>

</div>
""",
unsafe_allow_html=True
)

        try:

            # Retrieve relevant chunks
            retrieved_docs = (
                st.session_state.retriever.invoke(question)
            )

            # Generate response
            response = st.session_state.chain.invoke(question)

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
                "Gemini 2.5 Flash"
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

Gemini 2.5 Flash
"""
    )


# ----------------------------
# Footer
# ----------------------------

st.markdown("""
<div class="hero">

<h2>
🤖 Samsung AI Copilot
</h2>


<p>

Built with:

<br>

🚀 Streamlit

&nbsp;|&nbsp;

🦜 LangChain

&nbsp;|&nbsp;

⚡ Google Gemini 

&nbsp;|&nbsp;

🗄 ChromaDB


</p>


<p>

Created with ❤️ by ANUSHKA

</p>


</div>

""",
unsafe_allow_html=True)
