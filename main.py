"""
Simple main.py at the root level to help diagnose import issues
"""
from fastapi import FastAPI

app = FastAPI(title="Diagnostic App")

@app.get("/")
def read_root():
    return {"message": "Hello World from root main.py"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
