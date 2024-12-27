# config.py

MODEL_CONFIG = {
    "model_path": "path/to/your/finetuned/model",
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "gpu_memory_utilization": 0.90,
    "trust_remote_code": True,
}

SERVE_CONFIG = {
    "num_gpus_per_replica": 1,
    "min_replicas": 1,
    "max_replicas": 3,
    "target_num_ongoing_requests_per_replica": 50,
}

DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 512,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}
