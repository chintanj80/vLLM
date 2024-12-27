import requests

url = "http://localhost:8000/generate"
data = {
    "prompt": "Your prompt here",
    "sampling_params": {
        "temperature": 0.8,
        "max_tokens": 100
    }
}
response = requests.post(url, json=data)
print(response.json())
