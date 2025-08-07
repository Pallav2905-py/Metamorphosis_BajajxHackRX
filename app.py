import os
import json
import uuid
import logging
import requests
import torch
import asyncio
import hashlib
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
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
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 10))  # For parallel processing
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_DIR: str = os.getenv("CACHE_DIR", "document_cache")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))  # Batch size for embeddings

settings = Settings()

# Create cache directory
if settings.ENABLE_CACHE:
    os.makedirs(settings.CACHE_DIR, exist_ok=True)

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

# Setup embeddings with improved retrieval - PRELOAD AT STARTUP
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Preloading embedding model on device: {device}")

embedding_fn = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs={'device': device}
)
vector_store = PineconeVectorStore(index=index, embedding=embedding_fn, text_key="text")

logger.info("Embedding model preloaded successfully")

# Thread-safe Gemini client pool
class GeminiClientPool:
    def __init__(self, api_key: str, pool_size: int = 10):
        self.api_key = api_key
        self.clients = [genai.Client(api_key=api_key) for _ in range(pool_size)]
        self.lock = threading.Lock()
        self.available_clients = list(self.clients)
    
    def get_client(self):
        with self.lock:
            if self.available_clients:
                return self.available_clients.pop()
            else:
                return genai.Client(api_key=self.api_key)
    
    def return_client(self, client):
        with self.lock:
            if len(self.available_clients) < len(self.clients):
                self.available_clients.append(client)

# Initialize client pool
client_pool = GeminiClientPool(settings.GENAI_API_KEY, settings.MAX_WORKERS)

class StructuredAnswer(BaseModel):
    answer: str
    rationale: str
    sources: List[str]

class GeminiLLM(LLM):
    @property
    def _llm_type(self):
        return "genai"

    @property
    def _identifying_params(self):
        return {"model": settings.GENAI_MODEL}

    def _call(self, prompt: str, stop=None, max_retries: int = 3) -> str:
        """Call Gemini API with retry logic"""
        for attempt in range(max_retries):
            client = client_pool.get_client()
            try:
                response = client.models.generate_content(
                    model=settings.GENAI_MODEL,
                    contents=prompt,
                    config={
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "response_mime_type": "application/json",
                        "response_schema": StructuredAnswer.model_json_schema(),
                    }
                )
                return response.text.strip()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for Gemini API call")
                    return json.dumps({
                        "answer": "Unable to process question due to repeated API errors",
                        "rationale": f"API error after {max_retries} attempts: {str(e)}",
                        "sources": []
                    })
                else:
                    # Exponential backoff: wait 1s, 2s, 4s, etc.
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            finally:
                client_pool.return_client(client)

# Create a single LLM instance - PRELOAD AT STARTUP
logger.info("Preloading Gemini LLM...")
llm_instance = GeminiLLM()
logger.info("Gemini LLM preloaded successfully")

# Optimize retrieval with caching - PRELOAD CACHE STRUCTURES
retrieval_cache = {}
cache_lock = threading.Lock()

# Prompt template for explainable answers
prompt_template = PromptTemplate(
    template="""
You are an expert insurance policy analyst. Analyze the provided context carefully and answer the question with extreme precision.

IMPORTANT INSTRUCTIONS:
1. Look for EXACT numbers, percentages, time periods, and limits
2. Search through ALL context chunks for complete information
3. If you find partial information, state what you found and what's missing
4. For waiting periods, grace periods, limits - provide EXACT values from the document
5. Do not assume or generalize - stick to what's explicitly stated

Context:
{context}

Question:
{question}

Provide a comprehensive answer focusing on:
- Exact numbers, periods, percentages when available
- Specific conditions and requirements
- Any sub-limits or caps mentioned
- Exact definitions provided in the policy

Response format: JSON with answer, rationale, and sources fields.
""",
    input_variables=["context", "question"]
)


# Request and response schemas
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# FastAPI app
app = FastAPI(
    title="BajajxHackRX_Metamorphosis",
    description="LLM-powered contextual document QA with explainability",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Preload models and initialize cache on startup"""
    logger.info("🚀 Server startup - Models preloaded and ready!")
    logger.info(f"📊 Using device: {device}")
    logger.info(f"🧠 Embedding model: {settings.EMBEDDING_MODEL}")
    logger.info(f"🤖 Gemini model: {settings.GENAI_MODEL}")
    logger.info(f"⚡ Max workers: {settings.MAX_WORKERS}")
    logger.info(f"💾 Cache enabled: {settings.ENABLE_CACHE}")

# Load and split documents

def download_document(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download document from URL: {url}")
    file_path = "temp_doc.pdf"
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path

def get_document_hash(source: str) -> str:
    """Generate a hash for the document source"""
    if source.startswith("http"):
        # For URLs, hash the URL itself
        return hashlib.md5(source.encode()).hexdigest()
    else:
        # For local files, hash the file content
        with open(source, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

def is_document_cached(doc_hash: str) -> bool:
    """Check if document embeddings are already cached"""
    if not settings.ENABLE_CACHE:
        return False
    cache_file = os.path.join(settings.CACHE_DIR, f"{doc_hash}.json")
    return os.path.exists(cache_file)

def save_chunks_to_cache(doc_hash: str, chunks: List[str]):
    """Save document chunks to cache"""
    if not settings.ENABLE_CACHE:
        return
    cache_file = os.path.join(settings.CACHE_DIR, f"{doc_hash}.json")
    cache_data = {
        "hash": doc_hash,
        "chunks": chunks,
        "timestamp": asyncio.get_event_loop().time() if hasattr(asyncio, '_get_running_loop') else 0
    }
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False)
    logger.info(f"Cached {len(chunks)} chunks for document {doc_hash}")

def load_chunks_from_cache(doc_hash: str) -> List[str]:
    """Load document chunks from cache"""
    if not settings.ENABLE_CACHE:
        return []
    cache_file = os.path.join(settings.CACHE_DIR, f"{doc_hash}.json")
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        logger.info(f"Loaded {len(cache_data['chunks'])} chunks from cache for {doc_hash}")
        return cache_data['chunks']
    except:
        return []

def batch_embed_documents(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed documents in batches for better performance"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Process batch
        batch_embeddings = embedding_fn.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def optimized_text_splitting(docs, max_chunk_size: int = 200):  # Fixed token limit
    """Optimized text splitting with proper token limit for all-MiniLM-L6-v2"""
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=min(settings.CHUNK_OVERLAP, 50),  # Reduce overlap for smaller chunks
        model_name=settings.EMBEDDING_MODEL,
        tokens_per_chunk=max_chunk_size  # Keep within 256 token limit
    )
    return splitter.split_documents(docs)

def load_and_index(source: str) -> None:
    logger.info(f"Loading document: {source}")
    is_temp_file = False
    
    # Generate document hash for caching
    if source.startswith("http://") or source.startswith("https://"):
        doc_hash = get_document_hash(source)
        source = download_document(source)
        is_temp_file = True
    else:
        doc_hash = get_document_hash(source)
    
    # Check if document is already processed and cached
    if is_document_cached(doc_hash):
        logger.info(f"Document {doc_hash} found in cache, checking if already indexed...")
        # Quick check if vectors exist in Pinecone (simplified check)
        try:
            stats = vector_store.index.describe_index_stats()
            if stats.total_vector_count > 0:
                logger.info("Document appears to be already indexed, skipping re-indexing")
                if is_temp_file and os.path.exists(source):
                    os.remove(source)
                return
        except:
            pass

    try:
        # Load document
        if source.lower().endswith(".pdf"):
            loader = PyPDFLoader(source)
        elif source.lower().endswith(".docx") or source.lower().endswith(".doc"):
            loader = UnstructuredWordDocumentLoader(source)
        else:
            loader = UnstructuredEmailLoader(source)

        # Load and split documents with optimization
        logger.info("Loading document content...")
        docs = loader.load()
        
        logger.info("Splitting document into chunks...")
        chunks = optimized_text_splitting(docs)
        
        # Extract text content
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        # Save to cache
        save_chunks_to_cache(doc_hash, chunk_texts)
        
        # Batch embed documents
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks in batches...")
        embeddings = batch_embed_documents(chunk_texts, settings.BATCH_SIZE)
        
        # Batch upsert to Pinecone
        logger.info("Upserting vectors to Pinecone...")
        batch_size = 100  # Pinecone batch size
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_texts = chunk_texts[i:i + batch_size]
            
            records = []
            for text, emb in zip(batch_texts, batch_embeddings):
                records.append((str(uuid.uuid4()), emb, {"text": text}))
            
            vector_store.index.upsert(vectors=records)
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(embeddings) + batch_size - 1)//batch_size}")
        
        logger.info(f"Successfully indexed {len(chunks)} chunks.")

    except Exception as e:
        logger.error(f"Error loading document: {e}")
        raise
    finally:
        if is_temp_file and os.path.exists(source):
            os.remove(source)
            logger.info(f"Deleted temporary file: {source}")

def enhanced_retrieval(question: str, k: int = 15) -> str:  # Increased k for better context
    """Enhanced retrieval with caching"""
    
    # Check cache first
    cache_key = hashlib.md5(f"{question}_{k}".encode()).hexdigest()
    with cache_lock:
        if cache_key in retrieval_cache:
            logger.info(f"Retrieved cached context for question: {question[:50]}...")
            return retrieval_cache[cache_key]
    
    # Primary retrieval
    primary_docs = vector_store.similarity_search(question, k=k)
    
    # Simplified keyword expansion (reduce overhead)
    critical_keywords = ["grace period", "waiting period", "NCD", "Plan A", "sub-limit", "maternity", "cataract", "donor", "AYUSH", "hospital"]
    
    additional_docs = []
    for keyword in critical_keywords:
        if keyword.lower() in question.lower():
            additional_docs.extend(vector_store.similarity_search(f"{keyword}", k=3))
            break  # Only use first matching keyword to reduce calls
    
    # Combine and deduplicate
    all_docs = primary_docs + additional_docs
    seen_content = set()
    unique_docs = []
    
    for doc in all_docs:
        content = doc.page_content
        if content not in seen_content and len(content.strip()) > 20:  # Filter very short chunks
            seen_content.add(content)
            unique_docs.append(doc)
    
    # Generate context - increased limit for better accuracy
    context = "\n\n".join([doc.page_content for doc in unique_docs[:20]])
    
    # Cache the result
    with cache_lock:
        retrieval_cache[cache_key] = context
        # Keep cache size manageable
        if len(retrieval_cache) > 100:
            # Remove oldest entries (simplified LRU)
            keys_to_remove = list(retrieval_cache.keys())[:20]
            for key in keys_to_remove:
                del retrieval_cache[key]
    
    return context

def process_single_question(question: str, max_retries: int = 2) -> str:
    """Process a single question with enhanced retrieval and retry logic"""
    for attempt in range(max_retries):
        try:
            # Enhanced context retrieval
            context = enhanced_retrieval(question)
            
            if not context or len(context.strip()) < 50:
                logger.warning(f"Insufficient context retrieved for question: {question[:50]}...")
                # Try alternative retrieval with broader search
                context = vector_store.similarity_search(question, k=20)
                context = "\n\n".join([doc.page_content for doc in context])
            
            # Format prompt
            formatted_prompt = prompt_template.format(context=context, question=question)
            
            # Get LLM response using the global instance with retry
            raw_response = llm_instance._call(formatted_prompt, max_retries=3)
            
            logger.info(f"Raw response for '{question[:50]}...': {raw_response[:200]}...")
            
            # Parse response
            try:
                parsed = json.loads(raw_response)
                answer_text = parsed.get("answer", "")
                
                if not answer_text:
                    logger.warning(f"Empty answer received for question: {question[:50]}...")
                    if attempt < max_retries - 1:
                        continue
                    return "Unable to find answer in the document"
                
                return answer_text
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error for question '{question[:50]}...': {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying question processing (attempt {attempt + 2}/{max_retries})")
                    time.sleep(1)  # Brief pause before retry
                    continue
                return "Unable to process question due to response format error"
                
        except Exception as e:
            logger.error(f"Error processing question '{question[:50]}...' (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying question processing (attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
                continue
            return f"Error processing question: {str(e)}"
    
    return "Unable to process question after multiple attempts"

# Enhanced retrieval with fallback
def enhanced_retrieval_with_fallback(question: str, k: int = 15) -> str:
    """Enhanced retrieval with fallback strategies"""
    try:
        return enhanced_retrieval(question, k)
    except Exception as e:
        logger.warning(f"Primary retrieval failed: {e}, trying fallback...")
        try:
            # Fallback: simpler retrieval
            docs = vector_store.similarity_search(question, k=k)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e2:
            logger.error(f"Fallback retrieval also failed: {e2}")
            return "Unable to retrieve relevant context from document"

# Single endpoint for ingestion and query
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    try:
        # 1. Ingest document
        load_and_index(req.documents)
        
        # 2. Process questions in parallel with retry logic
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            # Submit all questions for parallel processing with retry
            future_to_question = {
                executor.submit(process_single_question, question, 3): question  # 3 retries per question
                for question in req.questions
            }
            
            # Collect results maintaining order
            question_results = {}
            completed_count = 0
            
            for future in as_completed(future_to_question):
                question = future_to_question[future]
                completed_count += 1
                logger.info(f"Completed {completed_count}/{len(req.questions)} questions")
                
                try:
                    result = future.result(timeout=60)  # 60 second timeout per question
                    question_results[question] = result
                except Exception as e:
                    logger.error(f"Final error in parallel processing for '{question[:50]}...': {e}")
                    question_results[question] = f"Processing failed after all retries: {str(e)}"
        
        # Maintain original question order
        answers = [question_results[question] for question in req.questions]
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"✅ Processed {len(req.questions)} questions in {processing_time:.2f} seconds")
        
        # Log success rate
        successful_answers = sum(1 for answer in answers if not answer.startswith("Error") and not answer.startswith("Processing failed") and not answer.startswith("Unable to process"))
        success_rate = (successful_answers / len(answers)) * 100
        logger.info(f"📊 Success rate: {success_rate:.1f}% ({successful_answers}/{len(answers)})")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Critical error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hello", response_model=QueryResponse)
async def hello_world():
    return QueryResponse(answers=["Hello, world!"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)