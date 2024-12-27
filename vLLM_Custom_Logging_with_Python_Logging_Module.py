import logging
from datetime import datetime
import json
from pathlib import Path

class LLMLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("vllm")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / f"vllm_{datetime.now():%Y%m%d}.log")
        fh.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(fh)
        
        # Performance metrics
        self.metrics = {
            "requests": 0,
            "tokens_generated": 0,
            "errors": 0,
            "total_latency": 0
        }
    
    def log_request(self, model_name: str, prompt: str, response: str, 
                   latency: float, tokens: int):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tokens_generated": tokens,
            "latency": latency
        }
        
        self.logger.info(json.dumps(log_entry))
        
        # Update metrics
        self.metrics["requests"] += 1
        self.metrics["tokens_generated"] += tokens
        self.metrics["total_latency"] += latency
    
    def log_error(self, model_name: str, error: str):
        self.logger.error(f"Model: {model_name}, Error: {error}")
        self.metrics["errors"] += 1
    
    def get_metrics(self):
        return {
            **self.metrics,
            "average_latency": (
                self.metrics["total_latency"] / self.metrics["requests"]
                if self.metrics["requests"] > 0 else 0
            )
        }
