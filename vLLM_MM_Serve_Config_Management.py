from pydantic_settings import BaseSettings
from typing import Dict

class Settings(BaseSettings):
    models_config: Dict[str, str] = {
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
    }
    max_batch_size: int = 32
    max_tokens: int = 2048
    gpu_memory_utilization: float = 0.85
    
    class Config:
        env_file = ".env"

settings = Settings()

# Use settings in your app
models = {
    name: LLM(
        model=path,
        gpu_memory_utilization=settings.gpu_memory_utilization,
    ) for name, path in settings.models_config.items()
}
