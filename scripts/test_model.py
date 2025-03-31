from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from cleaner.html_preprocessing import clean_html_format

# Path to your checkpoint
model_path = "./flan-t5-small-finetuned_doc/checkpoint-2808"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model = model.to("cpu")

# Input text to clean
#input_text = "<p><strong>IHRE AUFGABE</strong><br />Als Head of ITServicedesksindSie in 1. Linie fürfolgende Aufgaben zuständig:<br /><br /></ p><ul><li>Personelleund- fachliche Führung</li>"

input_text = """<p>Für unseren Kunden suchenwir Sie als erfahrenen Fassadenisoleur</p><p><br /><strong>Anforderungen:</strong><br />- langjährige Erfahrung als Fassadenisoleur<br />- Kompaktfassaden, Styropor, Mineralwolle sind kein Fremdwörter für Sie<br />- Führerschein Kat B von Vorteil<br />- deutsche Sprachkenntnisse <br />- zuverlässig undselbstständiges Arbeiten gewohnt</p><p> </p><p><strong>Arbeitsort:</strong><br />- Basel und Umgebung<br /><br />Haben wir Ihr Interessegeweckt? Dann freuen wiruns auf Ihre aussagekräftige Bewerbung per Email.</p>"""
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



