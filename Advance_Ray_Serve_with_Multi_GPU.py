from vllm.engine.ray import RayWorker
from vllm.engine.arg_utils import AsyncEngineArgs
import ray
from ray import serve
from typing import Dict, List

@serve.deployment(
    ray_actor_options={
        "num_gpus": 2  # Number of GPUs per model
    }
)
class ParallelModelServer:
    def __init__(self, model_name: str):
        self.engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=2,  # Split across 2 GPUs
            max_num_batched_tokens=4096,
            max_num_seqs=256,
            gpu_memory_utilization=0.85
        )
        
        # Initialize workers
        self.workers = [
            RayWorker.options(
                num_gpus=1
            ).remote(self.engine_args, rank=i)
            for i in range(self.engine_args.tensor_parallel_size)
        ]
    
    async def __call__(self, request: GenerateRequest) -> str:
        # Distribute work across GPUs
        futures = [
            worker.generate.remote([request.prompt])
            for worker in self.workers
        ]
        outputs = await ray.get(futures[0])  # Get from primary worker
        return outputs[0].outputs[0].text

# Deploy with multi-GPU configuration
app = serve.get_app()
model_deployments = {
    "llama2": ParallelModelServer.bind("meta-llama/Llama-2-7b-chat-hf"),
    "mistral": ParallelModelServer.bind("mistralai/Mistral-7B-Instruct-v0.1")
}
serve.run(model_deployments)
