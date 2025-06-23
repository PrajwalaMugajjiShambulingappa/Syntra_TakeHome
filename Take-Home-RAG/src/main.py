# src/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.prompt_generator import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    options: str

@app.post("/ask")
def ask_question(payload: Query):
    answer = run_pipeline(payload.question, payload.options)
    return {"answer": answer}

