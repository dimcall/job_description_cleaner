import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import csv

def normalize(text):
    '''Normalize text by removing extra spaces'''
    return " ".join(text.strip().split())

# Load model and tokenizer
model_path = "./flan-t5-small-finetuned_doc/checkpoint-2808"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Load noisy test set
test_file = "data/splits/dataset_test_noisy.jsonl"
with open(test_file, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

# Store predictions and targets
preds = []
targets = []
inputs_list = []

# Run inference
for example in tqdm(test_data):
    input_text = example["input"].strip()
    target_text = example["target"].strip()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    normalized_pred = normalize(decoded)
    normalized_target = normalize(target_text)

    inputs_list.append(input_text)
    preds.append(normalized_pred)
    targets.append(normalized_target)


# Compute metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

bleu_score = bleu.compute(predictions=preds, references=[[t] for t in targets])
rouge_score = rouge.compute(predictions=preds, references=targets, use_stemmer=True)

# Save all metrics
metrics_file = "metrics_noisy_test.csv"
with open(metrics_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Score"])
    writer.writerow(["BLEU", bleu_score["bleu"]])
    for k, v in rouge_score.items():
        writer.writerow([k, v])

# Save predictions per example
examples_file = "predictions_noisy_test.csv"
with open(examples_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Input", "Prediction", "Target"])
    for inp, pred, tgt in zip(inputs_list, preds, targets):
        writer.writerow([inp, pred, tgt])