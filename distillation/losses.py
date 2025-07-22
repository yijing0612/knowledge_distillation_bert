import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self,
                 alpha_ce: float = 0.5,
                 alpha_mse: float = 0.5,
                 temperature: float = 2.0):
         """
        Combined loss for classification + embedding distillation

        Args:
            alpha_ce: Weight for cross-entropy classification loss
            alpha_mse: Weight for embedding distillation loss
            temperature: Optional temperature for soft targets (not used here directly)
        """
         super().__init__()
         self.ce_loss = nn.CrossEntropyLoss()
         self.mse_loss = nn.MSELoss()
         self.alpha_ce = alpha_ce
         self.alpha_mse = alpha_mse
         self.temperature = temperature
        
    def forward(self,
                student_logits,
                student_embeddings,
                teacher_embeddings,
                labels):
        
        """
        Compute combined loss.

        Args:
            student_logits: [B, num_labels]
            student_embeddings: [B, hidden_dim]
            teacher_embeddings: [B, hidden_dim]
            labels: [B]

        Returns:
            Total loss (scalar)
        """

        loss_cls = self.ce_loss(student_logits, labels)
        loss_emb = self.mse_loss(student_embeddings, teacher_embeddings)

        total_loss = self.alpha_ce * loss_cls + self.alpha_mse * loss_emb
        return total_loss, loss_cls, loss_emb
