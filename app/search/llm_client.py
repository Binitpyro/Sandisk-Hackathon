import logging
import httpx
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.ollama_url = settings.ollama_url
        self.ollama_model = settings.ollama_model
        self.model = settings.gemini_model
        self._gemini_client: Optional[httpx.AsyncClient] = None
        self._ollama_client: Optional[httpx.AsyncClient] = None

    def _get_gemini_client(self) -> httpx.AsyncClient:
        """Lazy-create a reusable async HTTP client for Gemini."""
        if self._gemini_client is None or self._gemini_client.is_closed:
            self._gemini_client = httpx.AsyncClient(timeout=settings.gemini_timeout)
        return self._gemini_client

    def _get_ollama_client(self) -> httpx.AsyncClient:
        """Lazy-create a reusable async HTTP client for Ollama."""
        if self._ollama_client is None or self._ollama_client.is_closed:
            self._ollama_client = httpx.AsyncClient(timeout=settings.ollama_timeout)
        return self._ollama_client

    async def generate_answer(self, query: str, context: str) -> str:
        """Generates an answer using the provided context and query."""
        prompt = f"""
You are a personal memory assistant. Answer the user's question using ONLY the provided context snippets. 
If the answer is not in the context, say "I don't have enough information in your indexed files to answer this."

Context:
{context}

Question: 
{query}

Instructions:
1. Provide a concise and direct answer.
2. Cite the source files by their paths using [source_index] notation if possible.
3. Be professional and helpful.

Answer:
"""
        if self.api_key:
            return await self._call_gemini(prompt)
        else:
            return await self._call_ollama(prompt)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    async def _call_gemini(self, prompt: str) -> str:
        """Calls the Google Gemini API with retry logic for transient failures."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            client = self._get_gemini_client()
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                logger.error("Gemini API error (%d): <redacted>", response.status_code)
                return f"Gemini API returned error {response.status_code}. Please check your API key or connection."
            
            data = response.json()
            # Extract text from Gemini response structure
            try:
                if 'candidates' in data and data['candidates']:
                    return data['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError, TypeError):
                pass
            logger.error("Unexpected Gemini response structure (keys: %s)", list(data.keys()))
            return "Error parsing Gemini response."
        except httpx.HTTPError as e:
            logger.error("Gemini API call failed: %s", e)
            return "Error calling Gemini API. Please check your network connection."

    async def _call_ollama(self, prompt: str) -> str:
        """Fallback to local Ollama instance."""
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            client = self._get_ollama_client()
            response = await client.post(self.ollama_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response from Ollama.")
            else:
                return "Local LLM (Ollama) not responding. Please provide a GEMINI_API_KEY or start Ollama."
        except Exception as e:
            logger.warning("Ollama fallback failed: %s", e)
            return "No LLM available. Here are the top retrieved snippets from your files:"
