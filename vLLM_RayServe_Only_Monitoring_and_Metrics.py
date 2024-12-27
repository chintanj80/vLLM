from ray import serve
import time

@serve.deployment
class MonitoredvLLM:
    def __init__(self):
        self.llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
        self.total_requests = 0
        self.total_latency = 0
    
    async def __call__(self, prompt: str):
        start_time = time.time()
        result = self.llm.generate([prompt])
        latency = time.time() - start_time
        
        # Update metrics
        self.total_requests += 1
        self.total_latency += latency
        
        return {
            "result": result[0].outputs[0].text,
            "metrics": {
                "latency": latency,
                "avg_latency": self.total_latency / self.total_requests
            }
        }
