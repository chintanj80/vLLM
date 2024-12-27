from fastapi import FastAPI, HTTPException, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio

app = FastAPI()

@app.on_event("startup")
async def startup():
    redis = redis.asyncio.from_url("redis://localhost")
    await FastAPILimiter.init(redis)

@app.post("/generate")
@app.dependency(Depends(RateLimiter(times=10, seconds=60)))  # 10 requests per minute
async def generate_text(request: GenerateRequest):
    # Your generation code here
    pass
