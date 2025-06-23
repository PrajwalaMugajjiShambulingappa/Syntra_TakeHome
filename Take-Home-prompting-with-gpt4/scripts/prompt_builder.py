def format_qa(qa):
    q = f"Q: {qa['question']}\nOptions:"
    for key, val in qa['options'].items():
        q += f"\n{key}. {val}"
    a = f"Answer: {qa.get('answer', '')}"
    explanation = f"\nExplanation: {qa.get('explanation', '')}" if 'explanation' in qa else ''
    return f"{q}\n{a}{explanation}\n"

def build_prompt(few_shot_examples, test_question):
    prompt = "\n".join(format_qa(ex) for ex in few_shot_examples)
    prompt += "\nNow, answer the following question:\n"
    prompt += format_qa(test_question).replace("Answer: ", "").replace("Explanation: ", "")
    return prompt
