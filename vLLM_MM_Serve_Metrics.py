from prometheus_client import Counter, Histogram
import time

# Metrics
request_counter = Counter('model_requests_total', 'Total requests', ['model'])
latency_histogram = Histogram('model_latency_seconds', 'Request latency', ['model'])

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    start_time = time.time()
    
    try:
        llm = models[request.model_name]
        outputs = llm.generate([request.prompt])
        
        # Record metrics
        request_counter.labels(request.model_name).inc()
        latency_histogram.labels(request.model_name).observe(
            time.time() - start_time
        )
        
        return {"text": outputs[0].outputs[0].text}
    except Exception as e:
        # Error metrics
        error_counter.labels(request.model_name).inc()
        raise
