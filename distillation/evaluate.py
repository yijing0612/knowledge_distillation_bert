import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score

def evaluate_model(model, dataset, device, batch_size=32, use_student_keys=False):
    """
    Evaluate classification model (either student or HF LoRA model).
    
    Args:
        model: nn.Module or HuggingFace model
        dataset: DatasetDict
        device: torch device
        batch_size: int
        use_student_keys: bool – set to True if using 'student_input_ids'

    Returns:
        accuracy, classification report
    """
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, classification_report

    model.eval()
    model.to(device)

    loader = DataLoader(dataset["test"], batch_size=batch_size)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["student_input_ids" if use_student_keys else "input_ids"].to(device)
            attention_mask = batch["student_attention_mask" if use_student_keys else "attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Support both HuggingFace and custom outputs
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return acc, report, f1_micro, f1_macro
