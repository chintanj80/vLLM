from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.ray import RayWorker
import ray

# Initialize Ray
ray.init()

# Configure engine arguments
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=2,  # Split model across GPUs
    dtype="float16",
    gpu_memory_utilization=0.85
)

# Create Ray workers
workers = [
    RayWorker.options(
        num_gpus=1,
        resources={"worker": 1}
    ).remote(engine_args, rank=i)
    for i in range(engine_args.tensor_parallel_size)
]
