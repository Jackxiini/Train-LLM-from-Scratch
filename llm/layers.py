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
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
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
        in_type = X.dtype
        X_fp32 = X.to(torch.float32)
        X_norm = torch.mean(X_fp32**2, dim=-1, keepdim=True) + self.eps
        X_norm = X_norm.sqrt()
        result = self.weight * X_fp32 / X_norm
        return result.to(in_type)
    
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
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
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
        # Apply RoPE (Rotary Position Embedding) to input x:
        # 
        # Given x = [x0, x1, x2, x3, ..., x_{d-2}, x_{d-1}], we split it into d/2 pairs:
        #    [(x0, x1), (x2, x3), ..., (x_{d-2}, x_{d-1})]
        #
        # For each pair (x_{2k}, x_{2k+1}), RoPE applies a rotation:
        # [
        #   x_{2k} * cos(m * θ_k) - x_{2k+1} * sin(m * θ_k),
        #   x_{2k} * sin(m * θ_k) + x_{2k+1} * cos(m * θ_k)
        # ]
        #
        # Here:
        # - m is the position index in the sequence
        # - θ_k is the frequency base for the k-th dimension
        #   (typically: θ_k = 1 / θ^{2k/d})
        # - cos and sin are precomputed lookup tables of shape (max_seq_len, d/2)
        #
        # In code:
        # - cos/sin are expanded to match the full last dim via repeat_interleave
        # - (-x_odd, x_even) interleaving simulates imaginary rotation
        # - Final result is:
        #     x * cos + x_rotated * sin

        cos_theta_ik = self.cos_theta_ik[token_positions]
        sin_theta_ik = self.sin_theta_ik[token_positions]
        cos_theta_ik = torch.repeat_interleave(cos_theta_ik, 2, dim=-1)
        sin_theta_ik = torch.repeat_interleave(sin_theta_ik, 2, dim=-1)
        X_rotated = torch.stack([-X[..., 1::2], X[..., ::2]], dim=-1).reshape_as(X)
        return X * cos_theta_ik + X_rotated * sin_theta_ik

def softmax(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_X = torch.max(X, dim=dim, keepdim=True).values
    exp_X = torch.exp(X - max_X)
    return exp_X / torch.sum(exp_X, dim=dim, keepdim=True)

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
    attention_score /= math.sqrt(d_k)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, -torch.inf)
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
        self.RoPE = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
    
    def forward(self, X: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        X: (batch, seq_len, d_model)
        token_positions: (batch, seq_len) or None
        """
        B, L, _ = X.shape
        d_k = self.d_k

        # 1) 生成 token_positions
        if token_positions is None:
            # arange -> shape (L,)
            token_positions = torch.arange(L, device=X.device, dtype=torch.long)
            # unsqueeze & expand -> shape (B, L)
            token_positions = token_positions.unsqueeze(0).expand(B, L)
        # 再插入 head 维度 -> (B, 1, L)
        token_positions = token_positions.unsqueeze(1)

        # 2) QKV 投影 & reshape -> (B, heads, L, d_k)
        Q, K, V = self.qkv_proj(X).chunk(3, dim=-1)
        Q = einops.rearrange(Q, "b l (h dk) -> b h l dk", h=self.n_heads, dk=d_k)
        K = einops.rearrange(K, "b l (h dk) -> b h l dk", h=self.n_heads, dk=d_k)
        V = einops.rearrange(V, "b l (h dk) -> b h l dk", h=self.n_heads, dk=d_k)  # 用 d_k 代替 d_v

        # 3) 应用 RoPE
        Q = self.RoPE(Q, token_positions)  # RoPE 里会把 token_positions 广播到 head 维度
        K = self.RoPE(K, token_positions)

        # 4) causal mask -> (1,1,L,L)
        mask = torch.tril(torch.ones(L, L, device=X.device, dtype=torch.bool))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 5) attention & 合并 heads
        out = scaled_dot_product_attention(Q, K, V, mask=mask)    # (B,h,L,d_k)
        out = einops.rearrange(out, "b h l dk -> b l (h dk)")

        # 6) 最后投影
        return self.output_proj(out)
'''

class CausalMultiHeadAttentionRoPE(nn.Module):
    """
    Causal multi-headed self-attention layer with RoPE
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.

    Wq (h*d_k, d_model)
    Wk (h*d_k, d_model)
    Wv (h*d_v, d_model)
    Wo (d_model, h*d_v)

    let d_k = d_v = d_in = d_model / num_heads
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        RoPE: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        # query, key, value projections for all heads, but in a batch
        self.qkv_proj = Linear(d_model, d_model * 3, device=device, dtype=dtype)
        # output projection
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads

        if RoPE is None:
            assert max_seq_len is not None, "max_seq_len must be provided if RoPE is not provided"
            assert theta is not None, "theta must be provided if RoPE is not provided"
            self.RoPE = RotaryPositionalEmbedding(theta, self.head_size, max_seq_len, device=device)
        else:
            self.RoPE = RoPE

    def forward(
        self,
        in_features,
        token_positions: torch.Tensor = None,
    ):
        """
        Forward pass for the attention mechanism.

        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): input tensor
            token_positions (Int[Tensor, "... sequence_length"] | None): Optional tensor with the positions of the tokens

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Output tensor after applying attention.
        """
        *batch, sequence_length, d_model = in_features.shape
        assert d_model == self.d_model, f"Expected d_model {self.d_model}, but got {d_model}"
        qkv = self.qkv_proj(in_features)  # (..., sequence_length, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)  # (..., T, d_model) x 3

        q = rearrange(
            q,
            "batch sequence_length (num_heads head_size) -> batch num_heads sequence_length head_size",
            head_size=self.head_size,
            num_heads=self.num_heads,
        )
        k = rearrange(
            k,
            "batch sequence_length (num_heads head_size) -> batch num_heads sequence_length head_size",
            head_size=self.head_size,
            num_heads=self.num_heads,
        )
        v = rearrange(
            v,
            "batch sequence_length (num_heads head_size) -> batch num_heads sequence_length head_size",
            head_size=self.head_size,
            num_heads=self.num_heads,
        )

        if token_positions is None:
            # (1, 1, sequence_length)
            token_positions = torch.arange(sequence_length, device=in_features.device)
            token_positions = einx.rearrange("seq -> batch... seq", token_positions, batch=[1] * len(batch))

        # Duplicate token positions for each head
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        q = self.RoPE(q, token_positions)
        k = self.RoPE(k, token_positions)

        # (..., queries, keys)
        # boolean mask of shape (..., queries, keys), which in this case is
        # (..., sequence_length, sequence_length). assume len(batch) == 1
        causal_mask = torch.tril(
            torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=in_features.device)
        )
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, seq, seq]
        attention = scaled_dot_product_attention(q, k, v, causal_mask)  # (B, nh, seq_len, head_size)

        attention = rearrange(
            attention, "batch num_heads sequence_length head_size -> batch sequence_length (num_heads head_size)"
        ).contiguous()
        # output projection
        out = self.output_proj(attention)
        return out
'''