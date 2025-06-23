import json
import random

def load_qa_pairs(path="data/Q&A_pair.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_test_questions(path="data/test_questions.json"):
    with open(path, "r") as f:
        return json.load(f)

def sample_examples(qa_pairs, n=10):
    return random.sample(qa_pairs, n)

def sample_test_question(test_questions):
    return random.choice(test_questions)
