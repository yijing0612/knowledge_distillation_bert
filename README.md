# Knowledge Distillation & LoRA Fine-tuning for BERT

This project demonstrates two efficient approaches to compress or fine-tune BERT models for text classification tasks:

- ✅ **Knowledge Distillation**: Train a compact "student" model to mimic a large "teacher" model (e.g., `bert-large-uncased` ➜ `bert-base-uncased`).
- ✅ **LoRA Fine-tuning**: Use Low-Rank Adaptation (LoRA) to fine-tune BERT in a parameter-efficient way by injecting trainable adapters.

**Dataset Used**: [AG News](https://huggingface.co/datasets/ag_news)
