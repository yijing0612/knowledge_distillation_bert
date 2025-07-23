import argparse
import torch
from datasets import load_from_disk
from distillation.evaluate import evaluate_model
from models.student_model import StudentClassifier
from transformers import BertForSequenceClassification
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os

def load_student_model(checkpoint_path):
    model = StudentClassifier(model_name="bert-large-uncased")
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def load_peft_model(checkpoint_dir):
    base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.cls_layer_name = "classifier"
    return model

def timed_eval(model, dataset, device, use_student_keys=False):
    model.to(device)
    model.eval()
    start_time = time.time()
    acc, _, f1_micro, f1_macro = evaluate_model(model, dataset, device, use_student_keys=use_student_keys)
    end_time = time.time()
    return acc, f1_micro, f1_macro, end_time - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_checkpoint", type=str, default=None)
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--prefix_checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    if args.student_checkpoint:
        print(f"Evaluating Student Model: {args.student_checkpoint}")
        student_model = load_student_model(args.student_checkpoint)
        student_dataset = load_from_disk("data/tokenized_agnews_small")
        student_acc, student_micro, student_macro, student_time = timed_eval(student_model, student_dataset, device, use_student_keys=True)
        results.append({
            "Model": "Distilled (Student)",
            "Accuracy": student_acc,
            "F1 (Micro)": student_micro,
            "F1 (Macro)": student_macro,
            "Inference Time (s)": student_time
        })

    if args.lora_checkpoint:
        print(f"Evaluating LoRA Model: {args.lora_checkpoint}")
        lora_model = load_peft_model(args.lora_checkpoint)
        lora_dataset = load_from_disk("data/tokenized_lora")
        lora_acc, lora_micro, lora_macro, lora_time = timed_eval(lora_model, lora_dataset, device)
        results.append({
            "Model": "LoRA",
            "Accuracy": lora_acc,
            "F1 (Micro)": lora_micro,
            "F1 (Macro)": lora_macro,
            "Inference Time (s)": lora_time
        })

    if args.prefix_checkpoint:
        print(f"Evaluating Prefix Model: {args.prefix_checkpoint}")
        prefix_model = load_peft_model(args.prefix_checkpoint)
        prefix_dataset = load_from_disk("data/tokenized_lora")
        prefix_acc, prefix_micro, prefix_macro, prefix_time = timed_eval(prefix_model, prefix_dataset, device)
        results.append({
            "Model": "Prefix-Tuning",
            "Accuracy": prefix_acc,
            "F1 (Micro)": prefix_micro,
            "F1 (Macro)": prefix_macro,
            "Inference Time (s)": prefix_time
        })

    if not results:
        print("No models were provided. Please specify at least one checkpoint.")
        exit()

    df = pd.DataFrame(results)

    print("\n=== Model Comparison ===")
    print(df.to_markdown(index=False))

    # Plot
    sns.set(style="whitegrid")
    melted = df.melt(id_vars="Model", value_vars=["Accuracy", "F1 (Micro)", "F1 (Macro)"])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="Model", y="value", hue="variable")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
