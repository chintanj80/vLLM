from fastapi import FastAPI
from vllm import LLM
import os

app = FastAPI()

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_configs = {
            "llama2": "meta-llama/Llama-2-7b-chat-hf",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
            "codellama": "codellama/CodeLlama-7b-hf"
        }
    
    async def load_model(self, model_name: str):
        if model_name not in self.models:
            self.models[model_name] = LLM(
                model=self.model_configs[model_name]
            )
        return self.models[model_name]
    
    async def unload_model(self, model_name: str):
        if model_name in self.models:
            del self.models[model_name]
            return True
        return False

model_manager = ModelManager()

@app.post("/load_model/{model_name}")
async def load_model(model_name: str):
    await model_manager.load_model(model_name)
    return {"status": "success", "model": model_name}

@app.post("/unload_model/{model_name}")
async def unload_model(model_name: str):
    success = await model_manager.unload_model(model_name)
    return {"status": "success" if success else "model not loaded"}
