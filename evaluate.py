# evaluate.py
import torch
from datasets import load_from_disk
from distillation.evaluate import evaluate_model
import argparse
import os

def load_student_model(checkpoint_path):
    from models.student_model import StudentClassifier
    model = StudentClassifier(model_name="bert-large-uncased")
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def load_lora_model(checkpoint_dir):
    from transformers import BertForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["student", "lora"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model and dataset...")
    if args.model_type == "student":
        model = load_student_model(args.checkpoint)
        use_student_keys = True
    else:
        model = load_lora_model(args.checkpoint)
        use_student_keys = False

    dataset = load_from_disk(args.data_path)

    acc, report = evaluate_model(model, dataset, device, use_student_keys=use_student_keys)

    print(f"\nAccuracy: {acc:.4f}")
    print("Classification Report:\n", report)
