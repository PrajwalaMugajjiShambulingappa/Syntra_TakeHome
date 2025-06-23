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

'''samples = [
    {
        "question": "Olivia experienced persistent acid reflux despite medication. Her gastroenterologist scheduled an esophagogastroduodenoscopy (EGD) with biopsy. Which CPT code represents this procedure?",
        "options": "A. 43235\nB. 43239\nC. 43248\nD. 43245",
        "ground_truth": "B. 43239",
        "explanation": "CPT code 43239 represents an EGD with biopsy, single or multiple. Code 43235 is for EGD without biopsy, and the others involve different interventions such as dilation or stent placement."
    },
    {
        "question": "Matthew has been diagnosed with obstructive sleep apnea and requires a full-face CPAP mask for nighttime breathing assistance. Which HCPCS Level II code corresponds to a full-face CPAP mask?",
        "options": "A. A7030\nB. A7031\nC. A7032\nD. A7034",
        "ground_truth": "A. A7030",
        "explanation": "HCPCS code A7030 refers to a full-face CPAP mask, used for delivering positive airway pressure to both the nose and mouth. The other codes are for components such as cushions or nasal masks."
    },
    {
        "question": "After experiencing visual disturbances, Ava underwent a fluorescein angiography to evaluate the blood flow in her retina. Which CPT code represents this imaging procedure?",
        "options": "A. 92235\nB. 92250\nC. 92133\nD. 92240",
        "ground_truth": "A. 92235",
        "explanation": "CPT 92235 is specific to fluorescein angiography with interpretation and report. Code 92250 is for fundus photography, 92133 is for scanning optic nerve imaging, and 92240 is for indocyanine-green angiography."
    },
    {
        "question": "During a routine prenatal visit, Emily underwent a nuchal translucency ultrasound in the first trimester to assess the risk of chromosomal abnormalities. Which CPT code is appropriate for this procedure?",
        "options": "A. 76801\nB. 76813\nC. 76815\nD. 76817",
        "ground_truth": "B. 76813",
        "explanation": "CPT code 76813 is for nuchal translucency measurement via ultrasound in the first trimester. 76801 is a general first trimester ultrasound, while 76815 and 76817 refer to limited and transvaginal scans."
    },
    {
        "question": "A patient presents with severe right upper quadrant abdominal pain. The physician orders a hepatobiliary iminodiacetic acid (HIDA) scan. Which CPT code corresponds to this nuclear medicine imaging?",
        "options": "A. 78201\nB. 78205\nC. 78206\nD. 78215",
        "ground_truth": "C. 78206",
        "explanation": "CPT code 78206 is for hepatobiliary imaging with pharmacologic intervention (e.g., HIDA scan with CCK). The other codes refer to different gastrointestinal or liver studies but not with pharmacologic agent."
    }
]'''

samples_from_pdf = [
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
]


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
