import os
import json
import uuid
import logging
import requests
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders.email import UnstructuredEmailLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from google import genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings:
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")
    GENAI_API_KEY: str = os.getenv("GENAI_API_KEY")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    GENAI_MODEL: str = os.getenv("GENAI_MODEL")
    TOP_K: int = int(os.getenv("TOP_K", 5))

settings = Settings()

# Initialize client
pc = PineconeClient(api_key=settings.PINECONE_API_KEY)

# create index if not exists
if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT)
    )

index = pc.Index(settings.PINECONE_INDEX_NAME)

# Setup embeddings (HuggingFace - free) and vector store
embedding_fn = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
vector_store = PineconeVectorStore(index=index, embedding=embedding_fn, text_key="text")

# Initialize Gemini for inference (free-tier)
genai_client = genai.Client(api_key=settings.GENAI_API_KEY)

class GeminiLLM(LLM):
    @property
    def _llm_type(self):
        return "genai"

    @property
    def _identifying_params(self):
        return {"model": settings.GENAI_MODEL}

    def _call(self, prompt: str, stop=None) -> str:
        response = genai_client.models.generate_content(
            model=settings.GENAI_MODEL,
            contents=prompt
        )
        return response.text.strip()

# Instantiate LLM wrapper
llm = GeminiLLM()

# Prompt template for explainable answers
prompt_template = PromptTemplate(
    template="""
You are an expert document-analysis assistant.  Your *only* output
MUST be a valid JSON object with exactly these keys:
  • answer    - a string
  • rationale - a string
  • sources   - an array of short excerpts

DO NOT output ANYTHING else.  Absolutely no markdown, no bullet lists,
no prefixes, no apologies, no “Sure, here you go.”  If you are unable
to answer, respond with:
  { "answer": "", "rationale": "", "sources": [] }

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)


# Request and response schemas
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str
    rationale: str
    sources: List[str]

class QueryResponse(BaseModel):
    answers: List[Answer]

# FastAPI app
app = FastAPI(
    title="BajajxHackRX_Metamorphosis",
    description="LLM-powered contextual document QA with explainability",
    version="1.0.0"
)

# Load and split documents

def download_document(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download document from URL: {url}")
    file_path = "temp_doc.pdf"
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path

def load_and_index(source: str) -> None:
    logger.info(f"Loading document: {source}")
    is_temp_file = False

    if source.startswith("http://") or source.startswith("https://"):
        source = download_document(source)
        is_temp_file = True

    try:
        if source.lower().endswith(".pdf"):
            loader = UnstructuredPDFLoader(source)
        elif source.lower().endswith(".docx") or source.lower().endswith(".doc"):
            loader = UnstructuredWordDocumentLoader(source)
        else:
            loader = UnstructuredEmailLoader(source)

        docs = loader.load()
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=settings.CHUNK_OVERLAP,
            model_name=settings.EMBEDDING_MODEL
        )
        chunks = splitter.split_documents(docs)

        # Embed & upsert
        records = []
        embeddings = embedding_fn.embed_documents([chunk.page_content for chunk in chunks])
        for chunk, emb in zip(chunks, embeddings):
            records.append((str(uuid.uuid4()), emb, {"text": chunk.page_content}))
        vector_store.index.upsert(vectors=records)
        logger.info(f"Indexed {len(chunks)} chunks.")

    finally:
        if is_temp_file and os.path.exists(source):
            os.remove(source)
            logger.info(f"Deleted temporary file: {source}")

# Single endpoint for ingestion and query
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    try:
        # 1. Ingest document
        load_and_index(req.documents)

        # 2. Setup retrieval QA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": settings.TOP_K})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

        # 3. Answer questions
        answers = []
        for question in req.questions:
            result = qa_chain(question)
            raw = result.get("result", "")
            logger.info(f"LLM raw output:\n{raw!r}")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                raise HTTPException(502, detail=f"Malformed JSON from LLM for question “{question}”")
            logger.info(f"Parsed JSON keys: {list(parsed.keys())}")
            answers.append(Answer(
                question=question,
                answer=parsed.get("answer", ""),
                rationale=parsed.get("rationale", ""),
                sources=parsed.get("sources", []),
            ))
        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)