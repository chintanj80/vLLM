from vllm import LLM
import ray
from ray import serve
from typing import List

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=100
)
class vLLMDeployment:
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name)
    
    async def __call__(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt])
        return outputs[0].outputs[0].text

# Deploy the service
serve.run(vLLMDeployment.bind("meta-llama/Llama-2-7b-chat-hf"))
