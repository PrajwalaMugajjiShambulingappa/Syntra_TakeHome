from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas import evaluate
from src.indexing import build_vector_store
from src.prompt_generator import run_pipeline
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import json
from dotenv import load_dotenv
load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = build_vector_store()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def make_query(inputs):
    return inputs['question']

retrieval_chain = RunnableParallel({
    "context": RunnableLambda(make_query) | retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
    "options": RunnablePassthrough()
})

'''samples_from_pdf = [
    {
        "question": "During a regular checkup, Dr. Stevens discovered a suspicious lesion on the floor of Paul's mouth and decided to perform an excision. Which CPT code covers the excision of an oral lesion?",
        "options": "A. 40800\nB. 41105\nC. 41113\nD. 40804",
        "ground_truth": "C. 41113",
        "explanation": "CPT code 41113 refers to excision of lesion of floor of mouth; with primary closure. The other codes refer to different oral procedures or areas (e.g., lips or cheeks)."
    },
    {
        "question": "Riley had an overactive thyroid and was recommended for a thyroid lobectomy. Which CPT code pertains to this procedure?",
        "options": "A. 60210\nB. 60212\nC. 60220\nD. 60225",
        "ground_truth": "C. 60220",
        "explanation": "CPT 60220 describes a thyroid lobectomy, including isthmusectomy. The other codes cover more complex or different thyroid procedures like with limited neck dissection or total thyroidectomy."
    },
    {
        "question": "Clara's primary care doctor ordered an X-ray of her wrist after a fall. The radiologist took three views. Which CPT code pertains to this X-ray?",
        "options": "A. 73110\nB. 73100\nC. 73120\nD. 73115",
        "ground_truth": "A. 73110",
        "explanation": "CPT 73110 describes a radiologic examination of the wrist with 3 views. The others represent fewer views or different extremity imaging procedures."
    },
    {
        "question": "Megan had difficulty urinating. Her urologist performed a cystourethroscopy to diagnose the issue. Which CPT code describes this procedure?",
        "options": "A. 52000\nB. 50250\nC. 50200\nD. 50395",
        "ground_truth": "A. 52000",
        "explanation": "CPT code 52000 refers to cystourethroscopy (separate procedure). Other codes refer to kidney or ureter procedures, not bladder and urethra."
    },
    {
        "question": "James noticed a lump in his armpit. The doctor diagnosed it as an abscess and performed an incision and drainage. Which CPT code corresponds to this procedure?",
        "options": "A. 20002\nB. 20060\nC. 20000\nD. 20005",
        "ground_truth": "D. 20005",
        "explanation": "CPT 20005 is for incision and drainage of soft tissue abscess. Other codes relate to different regions or types of incisions (e.g., complicated or deeper tissue)."
    }
]'''

def load_formatted_questions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_questions = json.load(f)

    formatted_samples = []
    for q in raw_questions:
        options_str = "\n".join([f"{key}. {value}" for key, value in q["options"].items()])
        formatted_samples.append({
            "question": q["question"],
            "options": options_str,
            "ground_truth": f'{q["correct_answer"]["option"]}. {q["correct_answer"]["value"]}',
            "explanation": q["explanation"]
        })

    return formatted_samples

samples_from_pdf = load_formatted_questions("data/parsed_data/final_parsed_questions.json")

evaluation_data = []

for sample in samples_from_pdf:
    inputs = {
        "question": sample["question"],
        "options": sample["options"]
    }
    retrieval_output = retrieval_chain.invoke(inputs)
    contexts = retrieval_output["context"].split("\n\n")

    answer = run_pipeline(sample["question"], sample["options"])

    evaluation_data.append({
        "question": sample["question"],
        "answer": answer.strip(),
        "ground_truth": sample["ground_truth"],
        "contexts": contexts
    })

ragas_dataset = Dataset.from_list(evaluation_data)

results = evaluate(
    ragas_dataset,
    metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
    embeddings=embedding_model
)

print("\nRAGAS Evaluation Results:")
print(results)
