import os
import torch
from datasets import load_from_disk
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, TaskType, PrefixTuningConfig
torch.manual_seed(42)

# Load dataset
print("Loading tokenized dataset...")
dataset = load_from_disk("data/tokenized_lora_400")
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

# PEFT Prefix config
peft_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    num_virtual_tokens=30,
    encoder_hidden_size=model.config.hidden_size,
)

print("Injecting Prefix adapters...")
model = get_peft_model(model, peft_config)
model.cls_layer_name = "classifier"

# Define training arguments
training_args = TrainingArguments(
    output_dir="checkpoints/prefix400",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
    logging_dir="./logs/prefix400",
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
print("Starting Prefix fine-tuning...")
trainer.train()
print("Training complete. Model saved to:", training_args.output_dir)

eval_results = trainer.evaluate()
print(f"Eval results: {eval_results}")
