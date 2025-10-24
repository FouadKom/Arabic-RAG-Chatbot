# -----------------------------
# 🔹 Imports
# -----------------------------
import os
import requests
import tempfile
import asyncio
from typing import List
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# -----------------------------
# 🔹 Environment Variables Setup
# -----------------------------
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

if not os.environ.get("NO_GCE_CHECK"):
    os.environ["NO_GCE_CHECK"] = "true"

if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

if not os.environ.get("GRPC_ENABLE_FORK_SUPPORT"):
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

if not os.environ.get("NGROK_AUTHTOKEN"):
    os.environ["NGROK_AUTHTOKEN"] = os.environ.get("NGROK_AUTHTOKEN", "")

# -----------------------------
# 🔹 Initialize LLM
# -----------------------------
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# -----------------------------
# 🔹 Text Splitter
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# -----------------------------
# 🔹 Embedding Function
# ----------------------------
model_name = "OmarAlsaabi/e5-base-mlqa-finetuned-arabic-for-rag"
sentence_transformer_embeddings = SentenceTransformerEmbeddings(model_name=model_name)

# -----------------------------
# 🔹 Vector Store
# -----------------------------
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=sentence_transformer_embeddings,  # Make sure this is defined
    persist_directory="./chroma_langchain_db",
)

# -----------------------------
# 🔹 Document Utilities
# -----------------------------
def download_file_from_url(url: str, download_dir: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    filename = os.path.basename(urlparse(url).path) or "downloaded_file"
    file_path = os.path.join(download_dir, filename)
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path


def load_documents(source: str) -> List[Document]:
    documents = []
    temp_dir = tempfile.mkdtemp()

    if source.startswith("http://") or source.startswith("https://"):
        files_to_load = [download_file_from_url(source, temp_dir)]
    else:
        files_to_load = [
            os.path.join(source, f)
            for f in os.listdir(source)
            if os.path.isfile(os.path.join(source, f))
        ]

    for file_path in files_to_load:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            continue
        documents.extend(loader.load())

    return documents

# -----------------------------
# 🔹 Document Loading
# -----------------------------
url = "url_here"
documents = load_documents(url)
all_splits = text_splitter.split_documents(documents)


# -----------------------------
# 🔹 Chunks Embedding
# -----------------------------
_ = vector_store.add_documents(documents=all_splits)

# -----------------------------
# 🔹 Allow async in Colab
# -----------------------------
nest_asyncio.apply()

# -----------------------------
# 🔹 FastAPI Models
# -----------------------------
class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str

# -----------------------------
# 🔹 RAG Setup
# -----------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

template = """Answer the question in Arabic only, based solely on the following context and prior conversation.

Context:
{context}

Conversation History:
{chat_history}

Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

get_question = RunnableLambda(lambda x: x["question"])

rag_chain = (
    {
        "context": get_question | retriever | docs2str,
        "question": RunnablePassthrough(),
        "chat_history": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Session-based chat histories
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# -----------------------------
# 🔹 Initialize FastAPI
# -----------------------------
app = FastAPI(title="RAG API with Chat History & Streaming")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to the RAG API! Use POST /query or /query_stream to chat."}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    answer = rag_chain_with_history.invoke(
        {"question": request.question},
        config={"configurable": {"session_id": request.session_id}},
    )
    return QueryResponse(answer=answer)

@app.post("/query_stream")
async def query_rag_stream(request: QueryRequest):
    session_id = request.session_id
    question = request.question

    async def stream_generator():
        for chunk in rag_chain_with_history.stream(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        ):
            yield chunk + "\n"
            await asyncio.sleep(0)

    return StreamingResponse(stream_generator(), media_type="text/plain")

# -----------------------------
# 🔹 Start API via ngrok
# -----------------------------
public_url = ngrok.connect(8000)
print(f"🚀 FastAPI is live at: {public_url}")

config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
await server.serve()