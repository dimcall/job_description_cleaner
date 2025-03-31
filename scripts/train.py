# Cuda magic to avoid OOM errors
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import json
from datasets import Dataset
from clearml import Task
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, TrainerCallback
)
from transformers import AutoTokenizer
import evaluate

# Initialize ClearML Task
task = Task.init(project_name="job-description-cleaner", task_name="flan-t5-small-train")
logger = task.get_logger()

# Training Configuration
MODEL_NAME = "google/flan-t5-small"
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
EPOCHS = 4
LEARNING_RATE = 3e-4
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
OUTPUT_DIR = "./flan-t5-small-finetuned"

# Load dataset
train_data = [json.loads(line) for line in open("data/splits/dataset_train.jsonl")]
val_data = [json.loads(line) for line in open("data/splits/dataset_val.jsonl")]
test_data = [json.loads(line) for line in open("data/splits/dataset_test.jsonl")]

# Create datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
#val_dataset = val_dataset.select(range(4000)) 
test_dataset = Dataset.from_list(test_data)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token

def preprocess(data:dict)->dict:
    '''Helper function to tokenize inputs and targets
    data: dict, {"input": noisy, "target": cleaned text/original data}
    model_inputs: dict, {"input_ids": tokenized input, "attention_mask": attention mask, "labels": tokenized target}
    '''
    # Tokenize inputs
    model_inputs = tokenizer(
        data["input"], 
        max_length=MAX_INPUT_LENGTH, 
        padding="max_length", 
        truncation=True
    )
    # Tokenize targets
    labels = tokenizer(
        text_target=data["target"], 
        max_length=MAX_TARGET_LENGTH, 
        padding="max_length",
        truncation=True
    )
    # Assign labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
train_dataset = train_dataset.map(preprocess, batched=True, batch_size=1000)
val_dataset = val_dataset.map(preprocess, batched=True, batch_size=1000)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# Enable gradient checkpointing - reduce memory usage
model.gradient_checkpointing_enable()
# Set pad token id same as tokenizer pad token id
model.config.pad_token_id = tokenizer.pad_token_id

# Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
metric_accumulator = []

def compute_metrics(eval_preds):
    '''Compute BLEU, ROUGE and Copy Rate metrics
    '''
    # Unpack predictions and labels
    preds, labels = eval_preds

    # Convert to lists of ints if they are tensors
    preds = preds.tolist() if hasattr(preds, "tolist") else preds
    labels = labels.tolist() if hasattr(labels, "tolist") else labels

    def safe_decode(ids):
        '''Filter out invalid token ids that the tokenizer cannot decode'''
        return [id for id in ids if isinstance(id, int) and 0 <= id < tokenizer.vocab_size]

    # Decode predictions and labels e.g. preds = [[1, 4, 5]] -> decoded_preds = ["hello"]
    decoded_preds = tokenizer.batch_decode(
        [safe_decode(prediction) for prediction in preds],
        skip_special_tokens=True
    )
    decoded_labels = tokenizer.batch_decode(
        [safe_decode(label) for label in labels], 
        skip_special_tokens=True
    )

    # BLEU score - requires list of lists
    bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    # ROUGE score
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Exact match rate
    try:
        inputs = val_dataset[:len(decoded_preds)]["input"]
        # Exact match rate: 
        exact_match_rate = sum(p.strip() == i.strip() for p, i in zip(decoded_preds, inputs)) / len(decoded_preds)
    except Exception as e:
        print(f"Could not compute exact match rate: {e}")
        exact_match_rate = None

    # Store few random predictions for logging
    sample_logs = []
    for i in random.sample(range(len(decoded_preds)), min(5, len(decoded_preds))):
        sample_logs.append({
            "input": inputs[i],
            "prediction": decoded_preds[i],
            "target": decoded_labels[i]
        })
    # Store metrics
    result = {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
        "rougeLsum": rouge_score["rougeLsum"],
        "sample_predictions": sample_logs
    }

    if exact_match_rate is not None:
        result["exact_match_rate"] = exact_match_rate

    return result


# Callback for ClearML logging
class ClearMLLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        '''Log training metrics to ClearML - loss, learning rate, gradient norm'''
        logs = logs or {}
        step = state.global_step

        if "loss" in logs:
            logger.report_scalar(title="Train Loss", series="loss", value=logs["loss"], iteration=step)
        if "learning_rate" in logs:
            logger.report_scalar(title="Learning Rate", series="lr", value=logs["learning_rate"], iteration=step)
        if "grad_norm" in logs:
            logger.report_scalar(title="Gradient Norm", series="grad_norm", value=logs["grad_norm"], iteration=step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        '''Log evaluation metrics to ClearML - BLEU, ROUGE, Exact Match Rate'''
        step = state.global_step
        metrics = metrics or {}

        if "eval_loss" in metrics:
            logger.report_scalar(title="Eval Loss", series="loss", value=metrics["eval_loss"], iteration=step)

        if metric_accumulator:
            last_metrics = metric_accumulator[-1]

            logger.report_scalar(title="BLEU", series="Score", value=last_metrics["bleu"], iteration=step)
            logger.report_scalar(title="ROUGE-1", series="F1", value=last_metrics["rouge1"], iteration=step)
            logger.report_scalar(title="ROUGE-2", series="F1", value=last_metrics["rouge2"], iteration=step)
            logger.report_scalar(title="ROUGE-L", series="F1", value=last_metrics["rougeL"], iteration=step)
            logger.report_scalar(title="ROUGE-Lsum", series="F1", value=last_metrics["rougeLsum"], iteration=step)

            if "exact_match_rate" in last_metrics:
                logger.report_scalar(title="Copy Rate", series="rate", value=last_metrics["exact_match_rate"], iteration=step)
                
            if "sample_predictions" in last_metrics:
                combined = "\n\n".join(
                    f"Sample {idx+1}:\n"
                    f"Input: {sample['input']}\n"
                    f"Prediction: {sample['prediction']}\n"
                    f"Target: {sample['target']}"
                    for idx, sample in enumerate(last_metrics["sample_predictions"])
                )
                logger.report_text(combined, iteration=step)


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    eval_steps=500,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=1e-4,
    predict_with_generate=True,
    report_to=["clearml"],
    push_to_hub=False,
    fp16=False,
    bf16=True,
    gradient_accumulation_steps=4,
    dataloader_pin_memory=True,
    eval_accumulation_steps=1,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=1
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[ClearMLLoggingCallback()]
)

# Train
trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
