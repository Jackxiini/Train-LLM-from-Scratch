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
    m = inputs.amax(dim=-1, keepdim=True)
    log_sum_exp = m + (inputs - m).logsumexp(dim=-1, keepdim=True)
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    loss = log_sum_exp - target_logits
    return loss.mean()

def perplexity(inputs: Float[Tensor, "batch_size seq_len vocab_size"]) -> Tensor:
    return torch.exp(cross_entropy(inputs))



