
#libraries
from PIL import Image
import io
import os
import pytesseract
import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from dotenv import load_dotenv

load_dotenv()

file = r"C:\Users\HP\OneDrive\Desktop\CHATBOT\Tobi pdf Chatbot\CSR MODULES.pdf"
pdf = fitz.open(file)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

docs = []
for page_num , page in enumerate(pdf, start = 1):
  text = page.get_text().strip()
  if len(text) < 20:
    pix = page.get_pixmap(dpi = 400) #giving high pixels of 400
    img = Image.open(io.BytesIO(pix.tobytes("png")))  #converting text to image
    text = pytesseract.image_to_string(img).strip()  #extracting the text from image
  if text:
    docs.append(
        Document(page_content=text,
                 metadata={"source": os.path.basename(file), "page": page_num}
                 )
    ) #os.path.basename(file) trims the file path into pdf, two containers TEXT(page_content) and EXTRA INFO(metadata)


text_split = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)
chunks = text_split.split_documents(docs)


embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embedding)
faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

bm_25_retriever = BM25Retriever.from_documents(chunks)
bm_25_retriever.k = 4

ensemble = EnsembleRetriever(
    retrievers = [bm_25_retriever,faiss_retriever],
    weights = [0.5,0.5])

groq_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0.2
)

prompt = ChatPromptTemplate.from_template(
"""
You are Tobi, a helpful and reliable Al assistant.

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

rag_chain = (
    {
        "context": ensemble | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke(input("what is your question"))
print(answer)

while True:
    question = input("Ask a question (or type 'quit' to stop): ").strip()

    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    if not question:
        continue

    answer = rag_chain.invoke(question)
    print("\nAnswer:", answer)

    sources = ensemble.invoke(question)
    print("Sources: pages", sorted({d.metadata["page"] for d in sources}))
    print("-" * 50)

