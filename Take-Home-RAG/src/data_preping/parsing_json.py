import json
import re

def clean_text(text):
    text = text.replace("M e d i c a l C o d i n g A c e", "")
    return re.sub(r'\s+', ' ', text).strip()

def extract_data(json_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    text = data['content']

    # Extract questions
    q_pattern = re.compile(
        r"(\d+)\.\s(.*?)\nA\.\s(.*?)\nB\.\s(.*?)\nC\.\s(.*?)\nD\.\s(.*?)\n",
        re.DOTALL
    )
    questions = q_pattern.findall(text)

    # Extract answer key
    answer_key_pattern = re.compile(r"\[\s*\]\s*(\d+)\.\s([A-D])\.\s(.*?)\n")
    answer_key = {num: (opt, ans) for num, opt, ans in answer_key_pattern.findall(text)}

    # Extract explanations
    explanation_pattern = re.compile(
        r"(\d+)\.\s(.*?)\nAnswer:\s([A-D])\.\s(.*?)\nExplanation:\s(.*?)(?=\n\d+\.|\Z)",
        re.DOTALL
    )
    explanations = {num: clean_text(expl) for num, _, _, _, expl in explanation_pattern.findall(text)}

    parsed = []
    for num, question, a, b, c, d in questions:
        q_num = num.strip()
        correct_opt, correct_val = answer_key.get(q_num, ("", ""))
        explanation = explanations.get(q_num, "")
        parsed.append({
            "question": clean_text(question),
            "options": {
                "A": clean_text(a),
                "B": clean_text(b),
                "C": clean_text(c),
                "D": clean_text(d),
            },
            "correct_answer": {
                "option": correct_opt,
                "value": clean_text(correct_val)
            },
            "explanation": explanation
        })

    with open(output_path, "w") as f:
        json.dump(parsed, f, indent=2)

extract_data(
    "data/parsed_data/practice_test.json",
    "data/parsed_data/final_parsed_questions.json"
)
