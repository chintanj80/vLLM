@ray.remote(num_gpus=1)
class BatchInference:
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name)
    
    def process_batch(self, prompts: List[str]):
        outputs = self.llm.generate(prompts)
        return [output.outputs[0].text for output in outputs]

# Create multiple workers
workers = [BatchInference.remote("meta-llama/Llama-2-7b-chat-hf") 
          for _ in range(4)]

# Process batches in parallel
batches = [["prompt1", "prompt2"], ["prompt3", "prompt4"]]
results = ray.get([worker.process_batch.remote(batch) 
                  for worker, batch in zip(workers, batches)])
