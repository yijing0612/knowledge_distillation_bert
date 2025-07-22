import os
import torch
from datasets import load_from_disk
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
torch.manual_seed(42)

# Load dataset
print("Loading tokenized dataset...")
dataset = load_from_disk("data/tokenized_lora")
dataset = dataset["train"].train_test_split(test_size=0.2)

# Load tokenizer and model
print("Loading base model and tokenizer...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Configure LoRA (target BERT attention modules)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value", "key"],  
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

print("Injecting LoRA adapters...")
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="checkpoints/lora",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
    logging_dir="./logs/lora",
    save_strategy="epoch",
    logging_steps=10
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# Train
print("Starting LoRA fine-tuning...")
trainer.train()
print("Training complete. Model saved to:", training_args.output_dir)

eval_results = trainer.evaluate()
print(f"Eval results: {eval_results}")
