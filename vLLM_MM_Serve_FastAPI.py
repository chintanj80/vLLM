from fastapi import FastAPI, HTTPException
from typing import Dict, Optional
from vllm import LLM, SamplingParams
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    model_name: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

# Store models in a dictionary
models: Dict[str, LLM] = {
    "llama2-7b": LLM(model="meta-llama/Llama-2-7b-chat-hf"),
    "mistral-7b": LLM(model="mistralai/Mistral-7B-Instruct-v0.1"),
    "codellama": LLM(model="codellama/CodeLlama-7b-hf")
}

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    if request.model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    llm = models[request.model_name]
    outputs = llm.generate([request.prompt], sampling_params)
    
    return {
        "text": outputs[0].outputs[0].text,
        "model": request.model_name
    }

# Health check endpoint
@app.get("/models")
async def list_models():
    return {"available_models": list(models.keys())}
