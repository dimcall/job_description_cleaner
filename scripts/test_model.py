from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from cleaner.html_preprocessing import clean_html_format

# Path to your checkpoint
model_path = "./flan-t5-small-finetuned_doc/checkpoint-2808"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model = model.to("cpu")

# Input text to clean
input_text = "<p><strong>IHRE AUFGABE</strong><br />Als Head of ITServicedesk sind Sie in 1. Linie für folgende Aufgaben zuständig:<br /><br /></ p><ul><li>Personelleund-fachliche Führung</li>"
# Clean HTML and split into chunks
text_chunks = clean_html_format(input_text)

cleaned_text = []
for chunk in text_chunks:
    # Tokenize
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
    # Generate output
    outputs = model.generate(**inputs, max_length=256)
    # Decode
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_text.append(decoded)

print("Cleaned text chunks:", cleaned_text)



