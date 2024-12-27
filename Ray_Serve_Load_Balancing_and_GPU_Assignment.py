@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    num_replicas=4  # Create 4 replicas
)
class LoadBalancedServer:
    def __init__(self, model_configs: Dict[str, str]):
        # Determine GPU ID for this replica
        self.gpu_id = serve.get_replica_context().replica_id % torch.cuda.device_count()
        
        # Initialize models on specific GPU
        self.models = {}
        for name, path in model_configs.items():
            self.models[name] = LLM(
                model=path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                device=f"cuda:{self.gpu_id}"
            )
    
    async def __call__(self, request: GenerateRequest) -> str:
        model = self.models.get(request.model_name)
        if not model:
            raise ValueError(f"Model {request.model_name} not found")
        
        outputs = model.generate([request.prompt])
        return outputs[0].outputs[0].text

# Configuration for multiple models
model_configs = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}

serve.run(LoadBalancedServer.bind(model_configs))
