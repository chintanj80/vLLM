from vllm import LLM
import ray
from ray import serve
from typing import Dict, List
from pydantic import BaseModel

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    model_name: str
    max_tokens: int = 100
    temperature: float = 0.7

# Model deployment class
@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=100,
    autoscaling_config={"min_replicas": 1, "max_replicas": 4}
)
class ModelServer:
    def __init__(self, model_name: str):
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Adjust based on GPU count
            gpu_memory_utilization=0.85
        )
    
    async def __call__(self, request: GenerateRequest) -> str:
        outputs = self.model.generate([request.prompt])
        return outputs[0].outputs[0].text

# Deploy multiple models
app = serve.get_app()
llama2_deployment = ModelServer.bind("meta-llama/Llama-2-7b-chat-hf")
mistral_deployment = ModelServer.bind("mistralai/Mistral-7B-Instruct-v0.1")

serve.run({"llama2": llama2_deployment, "mistral": mistral_deployment})
