import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_k_tensor = torch.tensor(d_model // num_heads, dtype=torch.float32).cuda()
        
        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):



        # (batch, heads, Q_seq_len, d_k) x (batch, heads, d_k, K_seq_len) -> (batch, heads, Q_seq_len, K_seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(self.d_k_tensor)

        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_probs = torch.softmax(attention_scores, dim=-1)


        # (batch, heads, Q_seq_len, K_seq_len) x (batch, heads, K_seq_len, d_k) -> (batch, heads, Q_seq_len, d_k)
        output = torch.matmul(attention_probs, V)

        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        #(batch, seq_len, d_model) x (d_model, d_model) -> (batch, seq_len, d_model)
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        #(batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_K) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        #(batch, heads, seq_len, d_k)
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        #(batch, heads, seq_len, d_k) -> (batch, seq_len, heads, d_k) -> (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        #(batch, seq_len, d_model) x (d_model, d_model) -> (batch, seq_len, d_model)
        output = self.W_o(output)
        return output