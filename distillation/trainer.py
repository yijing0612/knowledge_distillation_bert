import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from distillation.losses import DistillationLoss
import os

def distill_train(student_model,
                  teacher_model,
                  dataset,
                  device,
                  num_epochs=3,
                  batch_size=32,
                  learning_rate=3e-5,
                  alpha_ce=0.5,
                  alpha_mse=0.5,
                  use_wandb=False):
    """
    Main distillation training loop

    Args:
        student_model: nn.Module
        teacher_model: SentenceTransformer or compatible encoder
        dataset: tokenized DatasetDict
        device: 'cuda' or 'cpu'
        num_epochs: int
        batch_size: int
        learning_rate: float
        alpha_ce: weight for CE loss
        alpha_mse: weight for MSE loss
        use_wandb: whether to log to Weights & Biases
    """

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["test"], batch_size=batch_size)

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval() # Freeze teacher

    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    criterion = DistillationLoss(alpha_ce=alpha_ce, alpha_mse=alpha_mse)

    if use_wandb:
        wandb.init(project="textclass-distill")
        wandb.watch(student_model)
    
    for epoch in range(num_epochs):
        student_model.train()
        total_loss, total_cls, total_emb = 0, 0, 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress:
            student_input_ids = batch["student_input_ids"].to(device)
            student_attention_mask = batch["student_attention_mask"].to(device)
            teacher_input_ids = batch["teacher_input_ids"].to(device)
            teacher_attention_mask = batch["teacher_attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                teacher_output = teacher_model({
                    'input_ids': teacher_input_ids,
                    'attention_mask': teacher_attention_mask
                })  # Returns embeddings directly
                teacher_embeddings = teacher_output['sentence_embedding'].to(device)

            student_logits, student_embeddings = student_model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask
            )

            loss, loss_cls, loss_emb = criterion(
                student_logits, student_embeddings, teacher_embeddings, labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_emb += loss_emb.item()

            progress.set_postfix({
                "loss": f"{total_loss:.2f}",
                "CE": f"{total_cls:.2f}",
                "MSE": f"{total_emb:.2f}"
            })

            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/loss_cls": loss_cls.item(),
                    "train/loss_emb": loss_emb.item()
                })

        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(train_loader):.4f}")

    if use_wandb:
        wandb.finish()
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(student_model.state_dict(), "checkpoints/student_model.pt")