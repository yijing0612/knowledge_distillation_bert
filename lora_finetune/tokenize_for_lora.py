import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.proprocess_agnews import load_and_tokenize_dataset

# Save a small subset for LoRA fine-tuning
load_and_tokenize_dataset(
    teacher_model_name="sentence-transformers/all-MiniLM-L6-v2",
    student_model_name="bert-base-uncased",
    max_length=128,
    save_path="data/tokenized_lora",
    small_subset=True,
    samples_per_class=100
)
