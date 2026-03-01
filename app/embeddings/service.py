import logging
import asyncio
import threading
from typing import List, Optional, TYPE_CHECKING
from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name or settings.embedding_model
        self.model: Optional["SentenceTransformer"] = None
        self._loading = False
        self._ready = threading.Event()

    def load_model(self) -> None:
        """Loads the embedding model (blocking)."""
        if self.model:
            self._ready.set()
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self._ready.set()
        logger.info("Model loaded successfully.")

    def load_model_background(self) -> None:
        """Starts model loading in a background thread (non-blocking)."""
        if self.model or self._loading:
            return
        self._loading = True
        thread = threading.Thread(target=self.load_model, daemon=True, name="emb-loader")
        thread.start()

    def wait_until_ready(self, timeout: float = 120) -> bool:
        """Block until the model is loaded. Returns True if ready."""
        return self._ready.wait(timeout=timeout)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def embed_texts(self, texts: List[str], batch_size: int = settings.embedding_batch_size) -> List[List[float]]:
        """Generates embeddings for a list of texts asynchronously."""
        if not self.model:
            if self._loading:
                await asyncio.get_running_loop().run_in_executor(
                    None, self.wait_until_ready
                )
            else:
                await asyncio.get_running_loop().run_in_executor(
                    None, self.load_model
                )
        if self.model is None:
            raise RuntimeError("Embedding model failed to load. Cannot generate embeddings.")

        loop = asyncio.get_running_loop()
        model = self.model  # capture for closure
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist(),
        )
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single query string."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]
