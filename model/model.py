import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModel(nn.Module):
  def __init__(self, embed_dim, num_heads):
      super(CrossAttentionModel, self).__init__()
      self.embed_dim = embed_dim
      self.num_heads = num_heads
      self.head_dim = embed_dim // num_heads

      assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

      # Learnable weight matrices
      self.query = nn.Linear(embed_dim, embed_dim)
      self.key = nn.Linear(embed_dim, embed_dim)
      self.value = nn.Linear(embed_dim, embed_dim)

      # Output projection
      self.out = nn.Linear(embed_dim, embed_dim)
      self.output_layer = nn.Linear(embed_dim, 2)

  def forward(self, x, context):
      """
      x: Query sequence tensor of shape (batch_size, seq_len_q, embed_dim)
      context: Context sequence tensor of shape (batch_size, seq_len_kv, embed_dim)
      """
      batch_size, seq_len_q, embed_dim = x.size()
      _, seq_len_kv, _ = context.size()

      # Generate query, key, and value matrices
      Q = self.query(x)  # Shape: (batch_size, seq_len_q, embed_dim)
      K = self.key(context)  # Shape: (batch_size, seq_len_kv, embed_dim)
      V = self.value(context)  # Shape: (batch_size, seq_len_kv, embed_dim)

      # Split into multiple heads
      Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
      K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
      V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

      # Compute scaled dot-product attention
      scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
      attention_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, num_heads, seq_len_q, seq_len_kv)
      attended_values = torch.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, seq_len_q, head_dim)

      # Concatenate the heads
      attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len_q, embed_dim)

      # Apply the output projection
      output = self.out(attended_values)
      output = self.output_layer(output)

      return output, attention_weights