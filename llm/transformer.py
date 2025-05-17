import torch
import torch.nn as nn
import einops
from llm.layers import *
from tokenizers import ByteLevelBPETokenizer

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int,
            d_ff: int,
            theta: float,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadAttentionRoPE(
            d_model=d_model,
            n_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, X: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        X shape: (..., sequence_length, d_model)
        token_positions shape: (..., sequence_length)
        out shape: (..., sequence_length, d_model)
        """
        X = self.ln1(X)
        X = self.attn(X, token_positions) + X
        X = self.ln2(X)
        X = self.ffn(X) + X

        return X

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device=None,
                 dtype=None):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.RoPE = RotaryPositionalEmbedding(rope_theta, d_model, context_length, device=device)
        self.transformer = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                max_seq_len=context_length,
                d_ff=d_ff,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens shape: (..., context_length)
        out shape: (..., context_length, vocab_size)
        """
        X = self.token_embeddings(tokens) 
        for layer in self.transformer:
            X = layer(X)
        X = self.ln_final(X)
        X = self.lm_head(X)
        return X

    @torch.no_grad()
    def generate(
            self,
            prompt_token_ids: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 0.0,
            eos_token_id: int = None,
            seed: int = None,
            device: str = None):
        
        assert top_k >= 0 and top_p >= 0.0
        assert temperature >= 0.0

        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        seq_len = prompt_token_ids.shape[1]
        if seq_len >= self.context_length:
            prompt_token_ids = prompt_token_ids[:, -self.context_length:]
        elif seq_len < self.context_length:
            prompt_token_ids = torch.cat([
                torch.zeros(prompt_token_ids.shape[0], self.context_length - seq_len, dtype=torch.int64, device=device),
                prompt_token_ids
            ], dim=1)

        prompt_token_ids = prompt_token_ids.to(device)
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        generated_token_ids = []
        current_prompt = prompt_token_ids
        while len(generated_token_ids) < max_new_tokens:
            logits = self.forward(current_prompt) # (1, context_length, vocab_size)
            new_token_logits = logits[0,-1] # (..., vocab_size)
            
            # Apply temperature scaling
            if temperature > 0:
                new_token_logits = new_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0 and top_k < self.vocab_size:
                top_k_probs, top_k_indices = torch.topk(new_token_logits, top_k, dim=-1)
                probs = softmax(top_k_probs, dim=-1)
                vocab_indices = top_k_indices
            else:
                probs = softmax(new_token_logits, dim=-1)
                vocab_indices = torch.arange(self.vocab_size, device=device)
            
            # Apply top-p (nucleus) filtering
            if top_p > 0.0:
                prob_sorted, indices_sorted = torch.sort(probs, dim=-1, descending=True)
                cum_probs = torch.cumsum(prob_sorted, dim=-1)
                threshold_indices = torch.where(cum_probs < top_p)[0]
                threshold_index = 0 if len(threshold_indices) == 0 else threshold_indices[-1]
                probs = prob_sorted[:threshold_index+1]
                top_p_indices = indices_sorted[:threshold_index+1]
                vocab_indices = vocab_indices[top_p_indices]
            
            # Sample from the filtered distribution
            sample_index = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).item()
            new_token = vocab_indices[sample_index]
            generated_token_ids.append(new_token)
            
            if eos_token_id is not None and new_token == eos_token_id:
                break

            # Update current prompt by shifting and adding new token
            new_token_tensor = torch.tensor([[new_token]], device=device)  # Shape: (1, 1)
            current_prompt = torch.cat([current_prompt[:, 1:], new_token_tensor], dim=1)

        return torch.tensor(generated_token_ids, device=device)

                
