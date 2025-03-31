from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from cleaner.html_preprocessing import clean_html_format

# Initialize FastAPI
app = FastAPI(title="Job Description Cleaner")

# Load model and tokenizer
model_path = "./flan-t5-small-finetuned_doc/checkpoint-2808"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

class HtmlInput(BaseModel):
    html: str

@app.post("/predict")
def predict(requests: List[HtmlInput]):
    results = []
    for req in requests:
        input_text = req.html
        text_chunks = clean_html_format(input_text)

        cleaned_text = []
        for chunk in text_chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs, max_length=512)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_text.append(decoded)

        results.append({"cleaned": cleaned_text})
    return results
