import torch
import torch.nn as nn
class Pyramidal(nn.Module):
    def __init__(self, proximal_dim, hidden_dim, distal_dim, dendrite_dim, k):
        super().__init__()
        self.proximal_dim = proximal_dim
        self.hidden_dim = hidden_dim
        self.distal_dim = distal_dim
        self.dendrite_dim = dendrite_dim
        self.k = k

        self.proximal = nn.Linear(proximal_dim, hidden_dim)
        self.distal = nn.Parameter(
            torch.randn(hidden_dim, dendrite_dim, distal_dim)
        )


    def forward(self, proximal_input, distal_input):
        proximal_output = self.proximal(proximal_input) # (batch_size, hidden_dim)
        distal_output = distal_input @ self.distal.T # (hidden_dim, batch_size, dendrite_dim)
        abs_max_idx = distal_output.abs().argmax(dim=-1, keepdim=True)
        distal_max = distal_output.gather(-1, abs_max_idx).squeeze(-1).T
        distal_modulate = torch.sigmoid(distal_max) # (batch_size, hidden_dim)
        result = proximal_output * distal_modulate # (batch_size, hidden_dim)

        #kWTA
        top_vals, top_idxs = torch.topk(result, k=self.k, dim=-1)
        mask = torch.zeros_like(result)
        mask.scatter_(dim=-1, index=top_idxs, value=1)
        result = result * mask

        return result