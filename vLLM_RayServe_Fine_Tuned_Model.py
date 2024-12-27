import ray
from ray import serve
from vllm import LLM, SamplingParams
from typing import Dict, List
import json
from fastapi import FastAPI

app = FastAPI()

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 3,
        "target_num_ongoing_requests_per_replica": 50,
    }
)
class LLMDeployment:
    def __init__(self):
        # Initialize the LLM
        self.llm = LLM(
            model="path/to/your/finetuned/model",  # Replace with your model path
            tensor_parallel_size=1,  # Adjust based on number of GPUs
            trust_remote_code=True,
            dtype="float16",  # Use half precision for efficiency
            gpu_memory_utilization=0.90,
        )
        
        # Default sampling parameters
        self.default_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )

    def generate_completion(self, prompt: str, params: Dict = None) -> Dict:
        """Generate completion for a given prompt"""
        # Use provided params or defaults
        sampling_params = self.default_params
        if params:
            sampling_params = SamplingParams(**params)
            
        outputs = self.llm.generate([prompt], sampling_params)
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        
        return {
            "generated_text": generated_text,
            "model": self.llm.model_config.model_path,
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
        }

    async def __call__(self, request):
        """Handle incoming requests"""
        body = await request.json()
        prompt = body.get("prompt")
        params = body.get("sampling_params", None)
        
        if not prompt:
            return {"error": "No prompt provided"}
            
        try:
            response = self.generate_completion(prompt, params)
            return response
        except Exception as e:
            return {"error": str(e)}

@app.post("/generate")
async def generate(request):
    handle = serve.get_deployment("llm").get_handle()
    response = await handle.remote(request)
    return response

def main():
    # Initialize Ray and Ray Serve
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    
    # Deploy the LLM service
    LLMDeployment.deploy()
    
    print("Service is ready at: http://localhost:8000/generate")

if __name__ == "__main__":
    main()
