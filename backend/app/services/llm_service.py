"""
LLM service using OpenRouter API.
Supports multiple models (Llama, Mistral, Gemma, DeepSeek, etc.)
"""
import json
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator

from app.core.config import get_settings
from app.core.logger import get_logger
from app.models.schemas import ChatMessage

logger = get_logger(__name__)
settings = get_settings()

# Available models on OpenRouter (free & paid tiers)
AVAILABLE_MODELS = [
    {
        "id": "meta-llama/llama-3.3-8b-instruct:free",
        "name": "Llama 3.3 8B Instruct (Free)",
        "provider": "Meta",
        "context_window": 131072,
        "free": True,
    },
    {
        "id": "meta-llama/llama-3.1-70b-instruct:free",
        "name": "Llama 3.1 70B Instruct (Free)",
        "provider": "Meta",
        "context_window": 131072,
        "free": True,
    },
    {
        "id": "mistralai/mistral-7b-instruct:free",
        "name": "Mistral 7B Instruct (Free)",
        "provider": "Mistral AI",
        "context_window": 32768,
        "free": True,
    },
    {
        "id": "google/gemma-3-4b-it:free",
        "name": "Gemma 3 4B Instruct (Free)",
        "provider": "Google",
        "context_window": 131072,
        "free": True,
    },
    {
        "id": "deepseek/deepseek-r1:free",
        "name": "DeepSeek R1 (Free)",
        "provider": "DeepSeek",
        "context_window": 163840,
        "free": True,
    },
    {
        "id": "qwen/qwen-2.5-72b-instruct:free",
        "name": "Qwen 2.5 72B Instruct (Free)",
        "provider": "Alibaba",
        "context_window": 131072,
        "free": True,
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "OpenAI",
        "context_window": 128000,
        "free": False,
    },
    {
        "id": "anthropic/claude-3-haiku",
        "name": "Claude 3 Haiku",
        "provider": "Anthropic",
        "context_window": 200000,
        "free": False,
    },
]


def build_rag_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    chat_history: Optional[List[ChatMessage]] = None,
) -> List[Dict[str, str]]:
    """
    Build the message list for the LLM using retrieved context.
    """
    # System prompt
    system_prompt = """You are SmartAI Notes — an intelligent document assistant.
Your job is to answer the user's questions **accurately and thoroughly** using ONLY the provided document context.

Guidelines:
- Base your answer strictly on the provided context. Do not hallucinate or add external knowledge unless explicitly asked.
- If the context doesn't contain enough information to answer, say: "I couldn't find enough information in the uploaded documents to answer this. Please try rephrasing or uploading more relevant files."
- Always cite the source document and page number when possible (e.g., "According to [filename], page 3...").
- Format your response clearly using markdown — use **bold**, bullet points, and headers when appropriate.
- If the question is a general greeting or off-topic, kindly redirect the user to ask document-related questions.
- Be concise but complete. Don't pad with filler text."""

    # Build context string
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        src = chunk.get("source", "unknown")
        page = chunk.get("page", 0)
        score = chunk.get("score", 0)
        content = chunk.get("content", "")
        page_info = f", page {page}" if page else ""
        context_parts.append(
            f"[Source {i}: {src}{page_info} | relevance: {score:.2f}]\n{content}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history (last 6 turns for context)
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

    # Final user message with context
    user_content = f"""## Retrieved Document Context:

{context_str}

---

## User Question:
{question}

Please answer the question based on the context above."""

    messages.append({"role": "user", "content": user_content})
    return messages


class LLMService:
    """Handles communication with OpenRouter's LLM API."""

    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = settings.OPENROUTER_BASE_URL
        self.default_model = settings.DEFAULT_LLM_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Devendra-Pudi/SmartAI_Notes",
            "X-Title": "SmartAI Notes",
        }

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM (non-streaming).
        Returns the full response dict.
        """
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            )

        if resp.status_code != 200:
            error_text = resp.text
            logger.error(f"OpenRouter API error {resp.status_code}: {error_text}")
            raise Exception(
                f"LLM API error (HTTP {resp.status_code}): {error_text[:300]}"
            )

        data = resp.json()
        return data

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from the LLM using SSE.
        Yields text tokens as they arrive.
        """
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                            token = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue

    def extract_answer(self, response_data: Dict[str, Any]) -> tuple[str, int]:
        """Extract answer text and token count from API response."""
        choices = response_data.get("choices", [])
        if not choices:
            return "No response generated.", 0

        answer = choices[0].get("message", {}).get("content", "").strip()
        usage = response_data.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        return answer, tokens

    def get_available_models(self) -> List[Dict]:
        return AVAILABLE_MODELS


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
