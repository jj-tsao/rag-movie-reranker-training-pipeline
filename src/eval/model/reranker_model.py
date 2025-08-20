import torch.nn as nn
from transformers import AutoModel

class RerankerModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_size=768, intermediate_size=768, dropout_prob=0.1, use_residual=True, use_layernorm=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.layernorm  = nn.LayerNorm(intermediate_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(intermediate_size, 1)
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_rep = outputs.last_hidden_state[:, 0]  # [CLS] token

        transformed = self.activation(self.linear1(cls_rep))

        if self.use_residual:
            # Add residual connections when dimensions match
            if transformed.size(-1) == cls_rep.size(-1):
                transformed = transformed + cls_rep
            # Skip residual connection if dimensions don't match (ptoentially, add a projection layer if needed in the future)
            else:
                pass 
        
        if self.use_layernorm:
            transformed = self.layernorm(transformed)

        transformed = self.dropout(transformed)   # Apply dropout after residual

        score = self.linear2(transformed).squeeze(-1)
        return score