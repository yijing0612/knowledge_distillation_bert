from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd
import os

def load_and_tokenize_dataset(teacher_model_name: str,
                              student_model_name: str,
                              max_length: int = 128,
                              save_path: str = None,
                              small_subset: bool = False,
                              samples_per_class: int = 100,
                              mode: str = "lora") -> DatasetDict:
    """
    Load and tokenize AG News dataset for LoRA fine-tuning or knowledge distillation.

    Args:
        teacher_model_name: Model name for teacher tokenizer
        student_model_name: Model name for student tokenizer
        max_length: Max sequence length
        save_path: Path to save tokenized dataset
        small_subset: Whether to use a small stratified sample
        samples_per_class: Number of samples per class (for small_subset)
        mode: "lora" or "distill"

    Returns:
        Tokenized DatasetDict
    """

    assert mode in ["lora", "distill"], "mode must be 'lora' or 'distill'"

    # Load AG News
    raw_datasets = load_dataset("ag_news")
    print("Loaded AG News dataset.")

    # Optional sampling
    if small_subset:
        print(f"Sampling {samples_per_class} per class")
        train_df = raw_datasets["train"].to_pandas()
        test_df = raw_datasets["test"].to_pandas()
        sampled_train_df = train_df.groupby("label").sample(n=samples_per_class, random_state=42)
        sampled_test_df = test_df.groupby("label").sample(n=samples_per_class, random_state=42)
        raw_datasets = DatasetDict({
            "train": Dataset.from_pandas(sampled_train_df, preserve_index=False),
            "test": Dataset.from_pandas(sampled_test_df, preserve_index=False)
        })

    # Load tokenizers
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    if mode == "distill":
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    # Tokenization function
    def tokenize(example):
        result = {
            "label": example["label"]
        }

        student = student_tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        result["input_ids"] = student["input_ids"]
        result["attention_mask"] = student["attention_mask"]

        if mode == "distill":
            teacher = teacher_tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            result["teacher_input_ids"] = teacher["input_ids"]
            result["teacher_attention_mask"] = teacher["attention_mask"]

        return result

    # Apply tokenizer
    tokenized = raw_datasets.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        tokenized.save_to_disk(save_path)
        print(f"Saved to {save_path}")

    return tokenized
