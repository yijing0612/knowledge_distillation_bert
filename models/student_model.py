from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn

class StudentClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 4, pooling: str = "cls", project_dim: int = 384):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.pooling = pooling
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

        # Projection layer to match teacher embedding size
        self.project = nn.Linear(self.encoder.config.hidden_size, project_dim)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = last_hidden_state[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * mask, 1)
            pooled = summed / torch.clamp(mask.sum(1), min=1e-9)
        else:
            raise ValueError("Invalid pooling method")

        # Project to teacher embedding size
        projected = self.project(pooled)

        if return_embedding:
            return projected

        logits = self.classifier(pooled)
        return logits, projected