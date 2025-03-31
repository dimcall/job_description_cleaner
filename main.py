from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from cleaner.html_preprocessing import clean_html_format

# Initialize FastAPI
app = FastAPI(title="Job Description Cleaner")

# Path to fine-tuned model checkpoint
model_path = "./flan-t5-small-finetuned_doc/checkpoint-2808"
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

class HtmlInput(BaseModel):
    html: str

@app.post("/predict")
def predict(requests: List[HtmlInput]):
    results = []
    for req in requests:
        # Clean HTML and split into chunks
        input_text = req.html
        text_chunks = clean_html_format(input_text)

        cleaned_text = []
        for chunk in text_chunks:
            # Tokenize
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
            # Generate output
            outputs = model.generate(**inputs, max_length=512)
            # Decode
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_text.append(decoded)

        results.append({"cleaned": cleaned_text})
    return results
