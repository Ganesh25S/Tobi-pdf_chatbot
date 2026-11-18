import streamlit as st
from PIL import Image
import io
import os
import hashlib
import pytesseract
import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

# ---------------- STREAMLIT PAGE ----------------
st.set_page_config(page_title="Tobi PDF Chatbot", page_icon="🤖")
st.title("🤖 Tobi PDF Chatbot")

# ---------------- PDF UPLOAD (replaces the hardcoded file path) ----------------
uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded is None:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

pdf_bytes = uploaded.getvalue()
file_hash = hashlib.md5(pdf_bytes).hexdigest()
file = uploaded.name

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- BUILD INDEX (runs only once per PDF) ----------------
if st.session_state.get("file_hash") != file_hash:
    with st.spinner("Reading PDF and building search index..."):

        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

        docs = []
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if len(text) < 20:
                pix = page.get_pixmap(dpi=400)  # giving high pixels of 400
                img = Image.open(io.BytesIO(pix.tobytes("png")))  # converting page to image
                text = pytesseract.image_to_string(img).strip()  # extracting the text from image
            if text:
                docs.append(
                    Document(page_content=text,
                             metadata={"source": os.path.basename(file), "page": page_num}
                             )
                )  # page_content = TEXT, metadata = EXTRA INFO

        if not docs:
            st.error("No text could be extracted from this PDF. Check the file and your Tesseract install.")
            st.stop()

        text_split = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_split.split_documents(docs)

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embedding)
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        bm_25_retriever = BM25Retriever.from_documents(chunks)
        bm_25_retriever.k = 4

        ensemble = EnsembleRetriever(
            retrievers=[bm_25_retriever, faiss_retriever],
            weights=[0.5, 0.5])

    # save in session so the index is not rebuilt on every question
    st.session_state.ensemble = ensemble
    st.session_state.file_hash = file_hash
    st.session_state.messages = []

ensemble = st.session_state.ensemble

# ---------------- LLM ----------------
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_template(
"""
You are Tobi, a helpful and reliable AI assistant.

If the user's message includes a greeting (hi, hello, hey, etc.),
respond to the greeting naturally and briefly first.


Then answer their actual question using the context below. Follow
any formatting or length instructions the user gives (e.g. "2 lines",
"in bullet points", "briefly").

If the question isn't covered by the context, say so clearly instead
of guessing — but still answer the greeting portion if there was one.

Rules:

1. For greetings and casual conversation, respond naturally.

2. If the question is covered by the document, use the provided context.

3. If the question is outside the document, answer using general knowledge.

4. Never invent or fabricate information.

5. If you are not confident, say "I'm not sure about that."

6. Do not claim information came from the document if it did not.

Context:
{context}

Question:
{question}

Answer:
""")


def format_docs(docs):
    return "\n\n".join(
        f"[Page {d.metadata['page']}] {d.page_content}" for d in docs
    )


# retrieval happens in the chat loop below (once per question), so the chain starts at the prompt
rag_chain = prompt | llm | StrOutputParser()

# ---------------- CHAT (replaces the while True input loop) ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

# show old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.caption("Sources: pages " + ", ".join(map(str, msg["pages"])))

# new question (replaces input("Ask a question..."))
question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sources = ensemble.invoke(question)
            answer = rag_chain.invoke({"context": format_docs(sources), "question": question})
            pages = sorted({d.metadata["page"] for d in sources})
        st.markdown(answer)
        st.caption("Sources: pages " + ", ".join(map(str, pages)))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "pages": pages}
    )
