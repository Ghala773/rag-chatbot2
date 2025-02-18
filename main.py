from fastapi import FastAPI
from model import generate_response

app = FastAPI()

@app.get("/")
def home():
    return {"Welcome to Aoun Bot"}

@app.post("/chat")
def chat(input_text: str):
    response = generate_response(input_text)
    return {"response": response}