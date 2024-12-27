from typing import Dict, List, Optional
import time
import json
from dataclasses import dataclass, asdict
import logging
import torch
from prometheus_client import Counter, Histogram, Gauge

@dataclass
class RequestMetrics:
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    status: str
    error: Optional[str] = None

class MetricsManager:
    def __init__(self, log_dir: str = "logs"):
        # Initialize logging
        self.logger = self._setup_logging(log_dir)
        
        # Initialize Prometheus metrics
        self.request_counter = Counter(
            'vllm_requests_total', 
            'Total requests',
            ['model', 'status']
        )
        self.latency_histogram = Histogram(
            'vllm_latency_seconds',
            'Request latency',
            ['model']
        )
        self.token_counter = Counter(
            'vllm_tokens_total',
            'Total tokens',
            ['model', 'type']
        )
        self.gpu_memory_gauge = Gauge(
            'vllm_gpu_memory_used_bytes',
            'GPU memory used',
            ['device']
        )
        
        # Initialize hardware monitor
        self.hw_monitor = HardwareMonitor()
    
    def _setup_logging(self, log_dir: str) -> logging.Logger:
        logger = logging.getLogger("vllm_metrics")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(f"{log_dir}/vllm_metrics.log")
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(handler)
        
        return logger
    
    def record_request(self, metrics: RequestMetrics):
        # Log to file
        self.logger.info(json.dumps(asdict(metrics)))
        
        # Update Prometheus metrics
        self.request_counter.labels(
            model=metrics.model_name,
            status=metrics.status
        ).inc()
        
        self.latency_histogram.labels(
            model=metrics.model_name
        ).observe(metrics.latency)
        
        self.token_counter.labels(
            model=metrics.model_name,
            type='prompt'
        ).inc(metrics.prompt_tokens)
        
        self.token_counter.labels(
            model=metrics.model_name,
            type='completion'
        ).inc(metrics.completion_tokens)
        
        # Update GPU metrics
        self._update_gpu_metrics()
    
    def _update_gpu_metrics(self):
        gpu_metrics = self.hw_monitor.get_gpu_metrics()
        for device_id, metrics in gpu_metrics.items():
            self.gpu_memory_gauge.labels(
                device=f"gpu_{device_id}"
            ).set(metrics.memory_used)

# Usage example
metrics_manager = MetricsManager()

async def generate_with_metrics(llm: LLM, prompt: str):
    start_time = time.time()
    
    try:
        outputs = llm.generate([prompt])
        latency = time.time() - start_time
        
        metrics = RequestMetrics(
            model_name=llm.model_name,
            prompt_tokens=len(llm.tokenize(prompt)),
            completion_tokens=len(outputs[0].outputs[0].token_ids),
            total_tokens=len(llm.tokenize(prompt)) + 
                         len(outputs[0].outputs[0].token_ids),
            latency=latency,
            status='success'
        )
        
    except Exception as e:
        metrics = RequestMetrics(
            model_name=llm.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency=time.time() - start_time,
            status='error',
            error=str(e)
        )
        raise
    finally:
        metrics_manager.record_request(metrics)
