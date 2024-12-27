from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    CompletionRequest, ChatCompletionRequest
)
from fastapi import FastAPI
import asyncio

app = FastAPI()

# Initialize models
models = {
    "llama2": AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model="meta-llama/Llama-2-7b-chat-hf")
    ),
    "mistral": AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model="mistralai/Mistral-7B-Instruct-v0.1")
    )
}

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    engine = models.get(request.model)
    if not engine:
        raise HTTPException(status_code=404, detail="Model not found")
    
    results = await engine.generate(request.prompt, request.sampling_params)
    return format_completion_response(results, request)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    engine = models.get(request.model)
    if not engine:
        raise HTTPException(status_code=404, detail="Model not found")
    
    results = await engine.generate(request.messages, request.sampling_params)
    return format_chat_response(results, request)
