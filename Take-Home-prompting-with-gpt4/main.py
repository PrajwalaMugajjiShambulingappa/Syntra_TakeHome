from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scripts.load_examples import load_qa_pairs, sample_examples
from scripts.prompt_builder import build_prompt
from scripts.gpt4_model import ask_gpt4

app = FastAPI()

# CORS setup for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    options: str

@app.post("/ask")
async def ask(request: AskRequest):
    few_shot_examples = sample_examples(load_qa_pairs(), n=5)
    test_question = {
        "question": request.question,
        "options": {
            line[0]: line[2:].strip()
            for line in request.options.split("\n")
            if line and line[1] in [".", ")"]
        },
    }

    prompt = build_prompt(few_shot_examples, test_question)
    answer = ask_gpt4(prompt)
    return {"answer": answer}
