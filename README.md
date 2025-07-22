# 🔬 Knowledge Distillation & LoRA Fine-tuning for BERT

This project demonstrates two efficient approaches to compress or fine-tune BERT models for text classification tasks:

- ✅ **Knowledge Distillation**: Train a compact "student" model to mimic a large "teacher" model (e.g., `bert-large-uncased` ➜ `bert-base-uncased`).
- ✅ **LoRA Fine-tuning**: Use Low-Rank Adaptation (LoRA) to fine-tune BERT in a parameter-efficient way by injecting trainable adapters.

**Dataset Used**: [AG News](https://huggingface.co/datasets/ag_news)

---

## 📁 Project Structure

knowledge_distillation_bert/
├── data/ # Tokenized datasets (generated)
├── distillation/ # Distillation training scripts
│ └── train_distill.py
│ └── evaluate.py
├── lora_finetune/ # LoRA fine-tuning scripts
│ └── train_lora.py
├── models/ # Student model definition
│ └── student_model.py
├── checkpoints/ # Saved model checkpoints
├── tokenize_dataset.py # Dataset preprocessing script
├── evaluate.py # Unified evaluation script
├── requirements.txt
└── README.md


---
