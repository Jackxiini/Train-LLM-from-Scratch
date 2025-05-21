from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None 
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * (grad * grad)
                
                lr_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                
                update = m / (torch.sqrt(v) + eps)
                p.data -= lr_t * update
                
                if weight_decay > 0:
                    p.data -= lr * weight_decay * p.data
                
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        return loss

def get_lr_cosine_schedule(it: int, 
                           max_learning_rate: float,
                           min_learning_rate: float,
                           warmup_iters: int,
                           cosine_cycle_iters: int):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    sum_of_squares = 0.0
    sigma = 1e-6
    for p in parameters:
        if p.grad is None:
            continue
        sum_of_squares += torch.sum(p.grad.data ** 2)
    
    total_norm = math.sqrt(sum_of_squares)
    if total_norm > max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= max_l2_norm / (total_norm + sigma)
    
    
    

                


