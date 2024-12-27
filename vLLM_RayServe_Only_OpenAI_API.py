from vllm.entrypoints.openai import OpenAIServingEndpoint
import ray
from ray import serve

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=100
)
class OpenAIEndpoint(OpenAIServingEndpoint):
    def __init__(self):
        super().__init__(
            model="meta-llama/Llama-2-7b-chat-hf",
            temperature=0.7,
            max_tokens=100
        )

# Deploy the endpoint
serve.run(OpenAIEndpoint.bind())
