from transformers import TrainingArguments

def get_training_args(output_dir="checkpoints/prefix", logging_dir="logs/prefix"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        save_strategy="epoch",
        logging_dir=logging_dir,
        logging_steps=50,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none"
    )
