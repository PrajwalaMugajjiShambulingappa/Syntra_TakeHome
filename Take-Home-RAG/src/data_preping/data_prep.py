import json
from pathlib import Path
from PyPDF2 import PdfReader

def pdf_to_json(pdf_path, json_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    json_data = {"content": full_text.strip()}
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

pdf_to_json("data/practice_test_answers (1) (1).pdf", "data/parsed_data/practice_test.json")
