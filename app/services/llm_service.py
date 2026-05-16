"""
LLM service — OpenRouter API with streaming support.
"""
import json
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator

from app.core.config import get_settings
from app.core.logger import get_logger
from app.models.schemas import ChatMessage

logger = get_logger(__name__)
settings = get_settings()

AVAILABLE_MODELS = [
    {"id": "meta-llama/llama-3.3-8b-instruct:free",   "name": "Llama 3.3 8B (Free)",    "provider": "Meta",       "free": True},
    {"id": "meta-llama/llama-3.1-70b-instruct:free",  "name": "Llama 3.1 70B (Free)",   "provider": "Meta",       "free": True},
    {"id": "mistralai/mistral-7b-instruct:free",       "name": "Mistral 7B (Free)",      "provider": "Mistral",    "free": True},
    {"id": "google/gemma-3-4b-it:free",                "name": "Gemma 3 4B (Free)",      "provider": "Google",     "free": True},
    {"id": "deepseek/deepseek-r1:free",                "name": "DeepSeek R1 (Free)",     "provider": "DeepSeek",   "free": True},
    {"id": "qwen/qwen-2.5-72b-instruct:free",          "name": "Qwen 2.5 72B (Free)",   "provider": "Alibaba",    "free": True},
    {"id": "openai/gpt-4o-mini",                       "name": "GPT-4o Mini",            "provider": "OpenAI",     "free": False},
    {"id": "anthropic/claude-3-haiku",                 "name": "Claude 3 Haiku",         "provider": "Anthropic",  "free": False},
]


def build_rag_prompt(question: str, context_chunks: List[Dict], chat_history: Optional[List[ChatMessage]] = None) -> List[Dict[str, str]]:
    system_prompt = """You are SmartAI Notes — an intelligent document Q&A assistant.

Rules:
- Answer ONLY based on the provided document context below.
- Always cite the source document and page (e.g. "According to [filename], page 3...").
- Use markdown formatting: **bold**, bullet points, headers where helpful.
- If context is insufficient, say: "I couldn't find enough information in the uploaded documents. Please upload more relevant files or rephrase your question."
- Be concise, accurate, and professional."""

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        src = chunk.get("source", "unknown")
        page = chunk.get("page", 0)
        score = chunk.get("score", 0)
        page_info = f", page {page}" if page else ""
        context_parts.append(f"[Source {i}: {src}{page_info} | relevance: {score:.2f}]\n{chunk.get('content','')}")

    context_str = "\n\n---\n\n".join(context_parts)
    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({
        "role": "user",
        "content": f"## Document Context:\n\n{context_str}\n\n---\n\n## Question:\n{question}"
    })
    return messages


class LLMService:
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = settings.OPENROUTER_BASE_URL
        self.default_model = settings.DEFAULT_LLM_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://huggingface.co/spaces/Devendra-Pudi/SmartAI-Notes",
            "X-Title": "SmartAI Notes",
        }

    async def generate(self, messages: List[Dict], model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 2048) -> Dict:
        payload = {"model": model or self.default_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload)
        if resp.status_code != 200:
            raise Exception(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    async def stream_generate(self, messages: List[Dict], model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 2048) -> AsyncGenerator[str, None]:
        payload = {"model": model or self.default_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", headers=self.headers, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                            token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue

    def extract_answer(self, data: Dict) -> tuple:
        choices = data.get("choices", [])
        answer = choices[0].get("message", {}).get("content", "").strip() if choices else "No response."
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return answer, tokens

    def get_available_models(self) -> List[Dict]:
        return AVAILABLE_MODELS


_llm: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm
    if _llm is None:
        _llm = LLMService()
    return _llm
