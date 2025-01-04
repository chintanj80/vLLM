import httpx
import asyncio
from fastapi import FastAPI, HTTPException

class VLLMApp:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None,
            "Content-Type": "application/json",
        }

    async def _make_request(self, endpoint, method="GET", json=None):
        url = f"{self.base_url}/{endpoint}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method, url, headers=self.headers, json=json
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_metrics(self):
        return await self._make_request("metrics")

    async def generate_completion(self, prompt, max_tokens=50):
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        return await self._make_request("completions", method="POST", json=payload)

    async def generate_chat_completion(self, messages, max_tokens=50):
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        return await self._make_request("chat/completions", method="POST", json=payload)

    async def generate_embeddings(self, input_text):
        payload = {
            "input": input_text,
        }
        return await self._make_request("embeddings", method="POST", json=payload)

app = FastAPI()
vllm_client = VLLMApp(base_url="http://localhost:8000")

@app.get("/metrics")
async def get_metrics():
    return await vllm_client.get_metrics()

@app.post("/completions")
async def generate_completion(prompt: str, max_tokens: int = 50):
    return await vllm_client.generate_completion(prompt, max_tokens)

@app.post("/chat/completions")
async def generate_chat_completion(messages: list[dict], max_tokens: int = 50):
    return await vllm_client.generate_chat_completion(messages, max_tokens)

@app.post("/embeddings")
async def generate_embeddings(input_text: str):
    return await vllm_client.generate_embeddings(input_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
