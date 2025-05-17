import torch
from torch import nn
import einops
from typing import Optional
import math
import einx
from einops import rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, device=device, dtype=dtype))
        mean = 0.0
        variance = 2/(self.in_features + self.out_features)
        std = variance**0.5
        nn.init.trunc_normal_(self.weight, mean, std, a=-3*std, b=3*std)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return einops.einsum(X, self.weight, "... in_features, out_features in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype))
        mean = 0.0
        std = 1
        nn.init.trunc_normal_(self.weight, mean, std, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert token_ids.dtype == torch.long, "token_ids must be a long tensor"
        return torch.nn.functional.embedding(token_ids, self.weight)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X shape: (batch_size, sequence_length, d_model)
        out shape: (batch_size, sequence_length, d_model)
        """
        dtype = X.dtype
        
        X_squared = X * X
        X_mean = X_squared.mean(dim=-1, keepdim=True)
        
        X_norm = torch.sqrt(X_mean + self.eps)
        
        normalized = X / X_norm
        result = self.weight * normalized
        
        return result.to(dtype)
    
class SiLU(nn.Module):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X * torch.sigmoid(X)

class SwiGLU(nn.Module):
    """
    SiLU = x * sigmoid(x)
    SwiGLU = W2*(SiLU(W1x)*W3x)
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        assert d_model % 64 == 0, "d_model should be a multiple of 64"
        if d_ff is None:
            d_ff = math.floor((8/3)*d_model)
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU()(self.w1(X))*self.w3(X))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        assert d_k % 2 == 0, "d_k should be even"
        k = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        theta_scale = 1.0 / (theta ** (2*k/d_k))
        seq_index = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        theta_ik = seq_index.view(len(seq_index), 1) * theta_scale.view(1, len(theta_scale))
        cos_theta_ik = torch.cos(theta_ik)
        sin_theta_ik = torch.sin(theta_ik)
        self.register_buffer("cos_theta_ik", cos_theta_ik, persistent=False)
        self.register_buffer("sin_theta_ik", sin_theta_ik, persistent=False)
    
    def forward(self, X: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        in_query_or_key shape: (..., sequence_length, d_k)
        token_positions shape: (..., sequence_length)
        out shape: (..., sequence_length, d_k)
        """
        in_type = X.dtype
        cos_theta_ik = self.cos_theta_ik[token_positions].to(in_type)
        sin_theta_ik = self.sin_theta_ik[token_positions].to(in_type)
        cos_theta_ik = torch.repeat_interleave(cos_theta_ik, 2, dim=-1)
        sin_theta_ik = torch.repeat_interleave(sin_theta_ik, 2, dim=-1)
        X_rotated = torch.stack([-X[..., 1::2], X[..., ::2]], dim=-1).reshape_as(X)
        return X * cos_theta_ik + X_rotated * sin_theta_ik

def softmax(inputs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax along the specified dimension.
    inputs shape: (..., dim)
    out shape: (..., dim)
    """
    in_type = inputs.dtype
    inputs_fp32 = inputs.to(torch.float32)
    m = inputs_fp32.amax(dim=dim, keepdim=True)
    exp_inputs = torch.exp(inputs_fp32 - m)
    return (exp_inputs / exp_inputs.sum(dim=dim, keepdim=True)).to(in_type)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Q shape: (..., queries, d_k)
    K shape: (..., keys, d_k)
    V shape: (..., keys, d_v)
    mask shape: (..., queries, keys)
    out shape: (..., queries, d_v)
    """
    d_k = K.shape[-1]
    attention_score = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    attention_score = attention_score / math.sqrt(d_k)  # Use division instead of /= to preserve dtype
    if mask is not None:
        attention_score = attention_score.masked_fill(~mask, float("-inf"))  # Use ~mask instead of mask == 0
    attention_weights = softmax(attention_score, dim=-1)
    output = einops.einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return output

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, device=None, dtype=None):
        super().__init__()
        self.qkv_proj = Linear(d_model, 3*d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X shape: (..., sequence_length, d_model)
        out shape: (..., sequence_length, d_model)
        """
        Q, K, V = self.qkv_proj(X).chunk(3, dim=-1)
        Q = einops.rearrange(Q, "... seq_len (n_heads d_k) -> ... n_heads seq_len d_k", n_heads = self.n_heads)
        K = einops.rearrange(K, "... seq_len (n_heads d_k) -> ... n_heads seq_len d_k", n_heads = self.n_heads)
        V = einops.rearrange(V, "... seq_len (n_heads d_v) -> ... n_heads seq_len d_v", n_heads = self.n_heads)
        causal_mask = torch.tril(torch.ones((X.shape[-2], X.shape[-2]), dtype=torch.bool))
        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        out = einops.rearrange(out, "... n_heads queries d_v -> ... queries (n_heads d_v)")
        return self.output_proj(out)


class CausalMultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.qkv_proj = Linear(d_model, 3*d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.RoPE = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device, dtype=dtype)
    
    def forward(self, X: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        X: (batch, seq_len, d_model)
        token_positions: (batch, seq_len) or None
        """
        B, L, _ = X.shape
        d_k = self.d_k

        # 1) Generate token positions
        if token_positions is None:
            token_positions = torch.arange(L, device=X.device, dtype=torch.long)
            token_positions = token_positions.unsqueeze(0).expand(B, L)
        token_positions = token_positions.unsqueeze(1)  # Add head dimension -> (B, 1, L)

        # 2) QKV projection & reshape -> (B, heads, L, d_k)
        Q, K, V = self.qkv_proj(X).chunk(3, dim=-1)
        Q = einops.rearrange(Q, "b l (h dk) -> b h l dk", h=self.n_heads, dk=d_k)
        K = einops.rearrange(K, "b l (h dk) -> b h l dk", h=self.n_heads, dk=d_k)
        V = einops.rearrange(V, "b l (h dk) -> b h l dk", h=self.n_heads, dk=d_k)

        # 3) Apply RoPE
        Q = self.RoPE(Q, token_positions)
        K = self.RoPE(K, token_positions)

        # 4) Create causal mask -> (1,1,L,L)
        mask = torch.tril(torch.ones(L, L, device=X.device, dtype=torch.bool))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 5) Attention & combine heads
        out = scaled_dot_product_attention(Q, K, V, mask=mask)    # (B,h,L,d_k)
        out = einops.rearrange(out, "b h l dk -> b l (h dk)")

        # 6) Final projection
        return self.output_proj(out)