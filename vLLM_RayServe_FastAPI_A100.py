import ray
from ray import serve
from vllm import LLM
from vllm.engine.ray import RayWorker
from vllm.engine.arg_utils import AsyncEngineArgs
from typing import Dict, List
from pydantic import BaseModel
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request model
class GenerateRequest(BaseModel):
    prompt: str
    model_name: str
    max_tokens: int = 100
    temperature: float = 0.7

# Initialize Ray cluster
ray.init(
    address="auto",  # Auto-detect the Ray cluster
    runtime_env={
        "pip": ["vllm", "transformers", "torch"],
    },
    _system_config={
        "distributed_gpu_num": 16,  # Total GPUs across all nodes
    }
)

@serve.deployment(
    ray_actor_options={
        "num_gpus": 8,  # GPUs per node
        "resources": {"node_type_a100": 1}  # Custom resource to ensure A100 placement
    },
    num_replicas=2  # One replica per node
)
class DistributedModelServer:
    def __init__(self, model_name: str):
        # Configure for 8-GPU tensor parallelism
        self.engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=8,  # Use all 8 GPUs on the node
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=8192,
            max_num_seqs=512,
            trust_remote_code=True,
            dtype="float16"  # Use FP16 for memory efficiency
        )
        
        # Initialize workers for each GPU
        self.workers = [
            RayWorker.options(
                num_gpus=1,
                resources={"node_type_a100": 0.125}  # 1/8 of node resource
            ).remote(self.engine_args, rank=i)
            for i in range(self.engine_args.tensor_parallel_size)
        ]
        
        logger.info(f"Initialized model {model_name} with {len(self.workers)} workers")
        
    async def __call__(self, request: GenerateRequest) -> Dict:
        try:
            start_time = time.time()
            
            # Distribute work across GPUs
            futures = [
                worker.generate.remote(
                    [request.prompt],
                    request.max_tokens,
                    request.temperature
                )
                for worker in self.workers
            ]
            
            # Get result from primary worker
            outputs = await ray.get(futures[0])
            
            return {
                "text": outputs[0].outputs[0].text,
                "metadata": {
                    "latency": time.time() - start_time,
                    "model": request.model_name,
                    "num_gpus_used": len(self.workers)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise serve.RayServeException(str(e))

# Model deployment configuration
MODEL_CONFIGS = {
    "llama2-70b": {
        "path": "meta-llama/Llama-2-70b-chat-hf",
        "gpu_memory_required": 80  # GB per GPU
    },
    "mixtral-8x7b": {
        "path": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gpu_memory_required": 70  # GB per GPU
    }
}

async def deploy_models():
    app = serve.get_app()
    deployments = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        deployment = DistributedModelServer.bind(config["path"])
        deployments[model_name] = deployment
        
        logger.info(f"Deploying {model_name}")
    
    serve.run(deployments)

# Resource monitoring
@serve.deployment
class ClusterMonitor:
    def __init__(self):
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "total_latency": 0
        }
    
    async def get_gpu_utilization(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            
            gpu_stats = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_stats.append({
                    "gpu_id": i,
                    "utilization": util.gpu,
                    "memory_used": memory.used / 1024**3,  # Convert to GB
                    "memory_total": memory.total / 1024**3
                })
            
            return gpu_stats
            
        except Exception as e:
            logger.error(f"Error getting GPU stats: {str(e)}")
            return []

    async def get_metrics(self):
        return {
            "cluster_metrics": {
                **self.metrics,
                "avg_latency": (
                    self.metrics["total_latency"] / self.metrics["requests"]
                    if self.metrics["requests"] > 0 else 0
                )
            },
            "gpu_metrics": await self.get_gpu_utilization()
        }

# Deployment script
if __name__ == "__main__":
    # Start Ray cluster monitoring
    monitor = ClusterMonitor.bind()
    serve.run({"monitor": monitor})
    
    # Deploy models
    ray.get(deploy_models.remote())
    
    logger.info("All models deployed successfully")

# Example usage:
async def test_deployment():
    handle = serve.get_deployment("llama2-70b").get_handle()
    response = await handle.remote({
        "prompt": "Hello, how are you?",
        "model_name": "llama2-70b",
        "max_tokens": 100
    })
    print(response)

# Health check endpoint
@serve.deployment
class HealthCheck:
    def __init__(self):
        self.monitor = ClusterMonitor.get_handle()
    
    async def __call__(self):
        metrics = await self.monitor.get_metrics.remote()
        gpu_utilization = metrics["gpu_metrics"]
        
        # Check if all GPUs are responsive
        healthy = len(gpu_utilization) == 16  # Expected 16 GPUs total
        
        return {
            "status": "healthy" if healthy else "degraded",
            "gpu_status": gpu_utilization,
            "timestamp": time.time()
        }

# Deploy health check
serve.run({"health": HealthCheck.bind()})