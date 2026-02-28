import os
import logging
import httpx
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.model = "gemini-flash-latest"

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

    async def _call_gemini(self, prompt: str) -> str:
        """Calls the Google Gemini API."""
        # Using v1beta as it often has broader support for newer models and aliases
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"Gemini API error ({response.status_code}): {error_detail}")
                    return f"Gemini API returned error {response.status_code}. Please check your API key or connection."
                
                data = response.json()
                # Extract text from Gemini response structure
                if 'candidates' in data and data['candidates']:
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    logger.error(f"Unexpected Gemini response structure: {data}")
                    return "Error parsing Gemini response."
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return "Error calling Gemini API. Please check your network connection."

    async def _call_ollama(self, prompt: str) -> str:
        """Fallback to local Ollama instance."""
        payload = {
            "model": "llama3", # Defaulting to llama3, can be configurable
            "prompt": prompt,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.ollama_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "No response from Ollama.")
                else:
                    return "Local LLM (Ollama) not responding. Please provide a GEMINI_API_KEY or start Ollama."
        except Exception as e:
            logger.warning(f"Ollama fallback failed: {e}")
            return "No LLM available. Here are the top retrieved snippets from your files:"
