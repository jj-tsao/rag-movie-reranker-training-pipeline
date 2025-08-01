import torch.nn as nn
from transformers import AutoModel

class RerankerModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_size=768, intermediate_size=768, dropout_prob=0.1, use_residual=True, use_layernorm=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.layernorm  = nn.LayerNorm(intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0]  # [CLS] token

        transformed = self.activation(self.linear1(cls_rep))

        if self.use_residual:
            if transformed.size(-1) == cls_rep.size(-1):  # Dimension check
                transformed = transformed + cls_rep
            else:
                pass # Skip residual connection if dimensions don't match
        
        if self.use_layernorm:
            transformed = self.layernorm(transformed)

        transformed = self.dropout(transformed)   # Apply dropout after residual

        score = self.linear2(transformed).squeeze(-1)
        return score