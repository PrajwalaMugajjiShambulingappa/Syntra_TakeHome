import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

def build_vector_store():
    # Load JSON content
    with open("data/parsed_data/final_parsed_questions.json", "r") as file:
        data = json.load(file)

    qa_texts = []
    for i, item in enumerate(data, 1):
        q = item["question"]
        opts = item["options"]
        ans = item["correct_answer"]
        expl = item.get("explanation", "")

        text = f"{i}. {q}\n"
        for key in ["A", "B", "C", "D"]:
            text += f"{key}. {opts.get(key, '')}\n"
        text += f"Answer: {ans['option']}. {ans['value']}\n"
        text += f"Explanation: {expl}\n"
        qa_texts.append(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.create_documents(qa_texts)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)
