@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=100
)
class DynamicModelServer:
    def __init__(self):
        self.models = {}
        self.model_configs = {
            "llama2": "meta-llama/Llama-2-7b-chat-hf",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
        }
        self.max_models = 2  # Maximum number of models in memory
    
    async def load_model(self, model_name: str):
        if model_name not in self.models:
            # Unload least recently used model if at capacity
            if len(self.models) >= self.max_models:
                lru_model = min(self.models.items(), key=lambda x: x[1]['last_used'])
                del self.models[lru_model[0]]
            
            # Load new model
            self.models[model_name] = {
                'model': LLM(model=self.model_configs[model_name]),
                'last_used': time.time()
            }
        
        self.models[model_name]['last_used'] = time.time()
        return self.models[model_name]['model']
    
    async def __call__(self, request: GenerateRequest) -> str:
        model = await self.load_model(request.model_name)
        outputs = model.generate([request.prompt])
        return outputs[0].outputs[0].text
