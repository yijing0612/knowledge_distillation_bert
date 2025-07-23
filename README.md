# Knowledge Distillation for BERT & Comparison with Fine Tuning

This project demonstrates two efficient approaches to compress or fine-tune BERT models for text classification tasks:

- **Knowledge Distillation**: Train a compact "student" model to mimic a large "teacher" model (`allMiniLM-L6-v2` ➜ `bert-base-uncased`).
- **Method Comparison**: Compare distilled model with fine-tuning version (Prefix Tuning & LoRA)

**Dataset Used**: [AG News](https://huggingface.co/datasets/ag_news)
