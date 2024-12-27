from ray.serve.config import AutoscalingConfig

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=4,
        target_num_ongoing_requests_per_replica=10
    )
)
class ScalablevLLM:
    def __init__(self):
        self.llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
