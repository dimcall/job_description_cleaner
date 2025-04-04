# 🧹 Job Description Cleaner

A multilingual sequence-to-sequence system that cleans job descriptions by removing HTML tags, fixing merged words, and formatting the content into readable chunks.

This project includes:

- 💾 Dataset generation from noisy HTML files
- 🧠 Fine-tuning a `flan-t5-small` model on synthetic noisy → clean examples
- 📈 Evaluation using BLEU, ROUGE, and exact match rate

---

## 📁 Project Structure

```text
.
├── cleaner/                   # HTML cleaning, noise generation, utilities
├── data/
│   ├── htmls/                 # Raw HTML job descriptions
│   └── splits/                # Train/val/test datasets (.jsonl)
├── flan-t5-small-finetuned/  # Trained model checkpoints
├── scripts/
│   ├── generate_dataset.py   # Generate noisy/clean dataset
│   ├── train.py              # Model training script
│   └── evaluate.py           # BLEU, ROUGE evaluation
├── main.py                   # FastAPI server for inference
├── client_interface.py       # Local script to query the model
└── README.md
```

## 🔧 Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 🏗️ Dataset Preparation

Generate the dataset (clean + noisy) from HTML job descriptions:

python scripts/generate_dataset.py

This creates:

dataset_train.jsonl

dataset_val.jsonl

dataset_test.jsonl

dataset_test_noisy.jsonl

## 🧠 Model Training

Fine-tune the model on noisy → clean examples:

```bash
python scripts/train.py
```

## 📊 Evaluation

Evaluate your trained model:

```bash
python scripts/test_model.py
```
