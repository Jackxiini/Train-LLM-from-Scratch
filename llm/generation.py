from contextlib import contextmanager
import torch
from llm.transformer import TransformerLM
from tokenizers import ByteLevelBPETokenizer

@contextmanager

def temporary_eval_model(model: TransformerLM):
    model.eval()
    try:
        yield
    finally:
        model.train()
        
def generateLLM(
        model: TransformerLM,
        tokenizer: ByteLevelBPETokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        eos_token_id: int = None,
        seed: int = None,
        device: str = None):
    
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Get token IDs from the Encoding object
    encoding = tokenizer.encode(prompt)
    prompt_token_ids = torch.tensor(encoding.ids, dtype=torch.int64, device=device).unsqueeze(0)  # Add batch dimension

    with temporary_eval_model(model), torch.inference_mode():
        generated_token_ids = model.generate(prompt_token_ids=prompt_token_ids,
                                             max_new_tokens=max_new_tokens,
                                             temperature=temperature,
                                             top_k=top_k,
                                             top_p=top_p,
                                             eos_token_id=eos_token_id,
                                             seed=seed,
                                             device=device)
        
        generated_text = tokenizer.decode(generated_token_ids.cpu().tolist())
        return generated_text
