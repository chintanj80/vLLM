import ray
# Get deployment metrics
handle = serve.get_deployment("llm").get_handle()
metrics = ray.get(handle.get_metrics.remote())
