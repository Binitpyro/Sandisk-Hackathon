from fastapi import FastAPI, Request, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import logging
from pathlib import Path

from app.storage.db import DatabaseManager
from app.indexing.service import IndexingService, progress
from app.embeddings.service import EmbeddingService
from app.vector_store.chroma_client import ChromaClient
from app.search.retrieval import full_rag
from app.search.llm_client import LLMClient
from app.insights.service import InsightsService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
db_manager = DatabaseManager()
embedding_service = EmbeddingService()
chroma_client = ChromaClient()
llm_client = LLMClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB on startup
    logger.info("Initializing database...")
    await db_manager.init_db()
    
    # Load ML model
    logger.info("Initializing ML model...")
    embedding_service.load_model()
    
    # Connect to Chroma
    logger.info("Initializing Chroma...")
    chroma_client.connect()
    
    yield
    # Cleanup
    logger.info("Shutting down...")
    await db_manager.close()

app = FastAPI(title="Personal Memory Assistant", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Dependencies
def get_db(): return db_manager
def get_emb(): return embedding_service
def get_chroma(): return chroma_client
def get_llm(): return llm_client

# Models
class IndexRequest(BaseModel):
    folders: List[str]

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/index/start")
async def index_start(
    request: IndexRequest, 
    background_tasks: BackgroundTasks, 
    db: DatabaseManager = Depends(get_db),
    emb: EmbeddingService = Depends(get_emb),
    chroma: ChromaClient = Depends(get_chroma)
):
    service = IndexingService(db, emb, chroma)
    background_tasks.add_task(service.index_folders, request.folders)
    return JSONResponse(status_code=202, content={"message": "Indexing started"})

@app.get("/index/status")
async def index_status(db: DatabaseManager = Depends(get_db)):
    async with db.conn.execute("SELECT COUNT(*) FROM files") as cursor:
        file_count = (await cursor.fetchone())[0]
    async with db.conn.execute("SELECT COUNT(*) FROM chunks") as cursor:
        chunk_count = (await cursor.fetchone())[0]
    
    percentage = 0
    if progress.total_files > 0:
        percentage = int((progress.processed_files / progress.total_files) * 100)
        
    return {
        "status": progress.status,
        "files_indexed": file_count,
        "chunks_indexed": chunk_count,
        "progress_percent": percentage
    }

@app.post("/query")
async def query(
    request: QueryRequest,
    db: DatabaseManager = Depends(get_db),
    emb: EmbeddingService = Depends(get_emb),
    chroma: ChromaClient = Depends(get_chroma),
    llm: LLMClient = Depends(get_llm)
):
    results = await full_rag(
        query=request.question,
        db=db,
        embedding_service=emb,
        chroma_client=chroma,
        llm_client=llm
    )
    
    # Track usage for insights
    for res in results["sources"]:
        await db.increment_usage_count(res["file_path"])
        
    return results

@app.get("/insights")
async def get_insights(db: DatabaseManager = Depends(get_db)):
    insights_svc = InsightsService(db)
    stats = await insights_svc.get_stats()
    return stats

@app.post("/demo/seed")
async def seed_demo(background_tasks: BackgroundTasks):
    """Creates some demo files and kicks off indexing."""
    demo_dir = Path("demo_data")
    demo_dir.mkdir(exist_ok=True)
    
    (demo_dir / "OS_project_feedback.txt").write_text(
        "TA Feedback on OS Project: The process scheduler implementation is correct, but your error handling in the fork system call needs improvement. 0.92/1.0"
    )
    (demo_dir / "internship_offer.md").write_text(
        "# Internship Offer\n\nDear Student, we are pleased to offer you a Software Engineering Internship at Future Corp. Starting date: June 1st, 2026. Salary: $8000/mo."
    )
    (demo_dir / "code_tips.md").write_text(
        "## Python Performance\nUse lists comprehensions instead of maps for small datasets. Always close DB connections."
    )
    
    # Trigger indexing
    service = IndexingService(db_manager, embedding_service, chroma_client)
    background_tasks.add_task(service.index_folders, [str(demo_dir.absolute())])
    
    return {"message": "Demo files created and indexing started.", "folder": str(demo_dir.absolute())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
