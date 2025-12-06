import requests

with open("OCR.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/chunk",
        files={"file": f},
        data={"chunk_size": 1024, "overlap": 100}
    )
    
result = response.json()
print(f"總共分割成 {result['total_chunks']} 個 chunks")