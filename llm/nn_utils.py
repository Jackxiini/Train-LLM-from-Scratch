from typing import Tuple, Optional
from jaxtyping import Float, Int

import torch
from torch import Tensor
from llm.layers import softmax

def log_softmax(inputs: Float[Tensor, "batch_size seq_len vocab_size"]) -> Float[Tensor, "batch_size seq_len vocab_size"]:
    return torch.log(softmax(inputs))

def cross_entropy(inputs: Float[Tensor, "batch_size seq_len vocab_size"],
                 targets: Int[Tensor, "batch_size seq_len"]) -> Tensor:
    """
    Compute the cross-entropy loss for a sequence of logits and targets.
    inputs shape: (batch_size, seq_len, vocab_size)
    targets shape: (batch_size, seq_len)
    """
    # Normalize logits for numerical stability
    logits_max = inputs.amax(dim=-1, keepdim=True)
    logits = inputs - logits_max  # Subtract max for numerical stability
    
    # Compute log probabilities using log_softmax
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    
    # Gather target log probabilities
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1))
    
    # Compute loss
    loss = -target_log_probs.mean()
    return loss

def perplexity(loss: Tensor) -> Tensor:
    """
    Compute perplexity from cross-entropy loss.
    Clamp loss to avoid overflow when exponentiating.
    """
    return torch.exp(torch.clamp(loss, max=100.0))  # Clamp to avoid overflow



