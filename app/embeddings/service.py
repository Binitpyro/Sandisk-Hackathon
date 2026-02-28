import logging
import asyncio
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Loads the embedding model (blocking, so call during startup)."""
        if not self.model:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully.")

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates embeddings for a list of texts asynchronously."""
        if not self.model:
            self.load_model()
        
        # sentence-transformers encode is blocking, run in executor
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True).tolist()
        )
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single query string."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]
