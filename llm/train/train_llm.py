import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import numpy as np
import numpy.typing as npt
import os
from typing import Union, Tuple
from llm.data import get_batch, random_training_iterator, SequentialValidationDataset
from llm.nn_utils import cross_entropy
from llm.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from llm.transformer import TransformerLM
from llm.serialization import save_checkpoint, load_checkpoint
from llm.tokenization import Tokenizer
from llm.generation import generateLLM, temporary_eval_model
import shutil
from loguru import logger
import torch
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
from collections.abc import Iterator
from torch.utils.data import IterableDataset
import logging
from torch.amp import autocast

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

VAL_PROMPTS = [
    "Once upon a time there was a big man named Jack. He was a good",
    "Once upon a time, there was a pretty girl named Lily. She hated to swin",
]

def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
def delete_output_dir(output_dir):
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")

def run_generation(
        model: TransformerLM,
        tokenizer: ByteLevelBPETokenizer,
        val_prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.0,
        seed: int = 42,
        device: str = "cuda:0"):
    
    with temporary_eval_model(model), torch.inference_mode():
        eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        generated_text = generateLLM(model, tokenizer, val_prompt, max_new_tokens, temperature, top_k, top_p, eos_token_id, seed, device)
        return generated_text

def train_LLM(
    model: TransformerLM,
    tokenizer: Tokenizer,
    num_iters: int,
    device: str,
    train_dataloader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    val_dataloader: Union[IterableDataset, None],
    out: str,
    checkpoint_interval: int,
    log_interval: int,
    log_wandb: bool,
    args: argparse.Namespace
) -> None:
    """
    Train a language model using the provided dataloaders.
    
    Args:
        model: The transformer model to train
        optimizer: Optimizer for training
        num_iters: Number of training iterations
        device: Device to train on (e.g. 'cuda:0' or 'cpu')
        train_dataloader: Iterator yielding (input, target) tensor pairs for training
        val_dataloader: Dataset for validation, or None if no validation
        out: Directory to save model checkpoints
        checkpoint_interval: How often to save checkpoints
        log_interval: How often to log training metrics
        log_wandb: Whether to log to Weights & Biases
    """
    # Handle output directory creation
    if os.path.exists(out):
        if os.path.isfile(out):
            # If it's a file, create a directory with a different name
            out = f"{out}_dir"
            logger.info(f"Output path was a file, using directory instead: {out}")
    os.makedirs(out, exist_ok=True)
    logger.info(f"Using output directory: {out}")

    logger.info(f"Starting training with device: {device}, precision: {args.precision}")
    logger.info(f"Model parameters: d_model={args.d_model}, num_layers={args.num_layers}, num_heads={args.num_heads}")
    logger.info(f"Training parameters: batch_size={args.batch_size}, context_length={args.context_length}, num_iters={num_iters}")
    logger.info(f"Optimizer parameters: max_lr={args.max_lr}, min_lr={args.min_lr}, warmup_iters={args.warmup_iters}")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if args.precision == "bfloat16":
        precision = torch.bfloat16
        logger.info("Using bfloat16 precision")
    elif args.precision == "float16":
        precision = torch.float16
        logger.info("Using float16 precision")
    else:
        precision = torch.float32
        logger.info("Using float32 precision")
        
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas = (args.beta1, args.beta2),
        eps=1e-6 if args.precision == "bfloat16" else 1e-8,
        weight_decay=args.weight_decay,
    )
    # Convert device string to torch.device object
    device = torch.device(device)
    
    start_iter = 0
    if args.resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        start_iter = load_checkpoint(model, optimizer, args.resume_from_checkpoint)
        logger.info(f"Resumed from iteration {start_iter}")

    model.train()
    try:
        import triton
        logger.info("Triton is available, compiling model with torch.compile...")
        compiled_model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
        logger.info("Model compilation successful")
    except ImportError:
        logger.warning("Triton is not available. Running without torch.compile (this may be slower)")
        compiled_model = model
    except Exception as e:
        logger.warning(f"Model compilation failed: {str(e)}. Falling back to eager mode.")
        compiled_model = model

    best_loss = float("inf")
    logger.info(f"Starting training loop from iteration {start_iter}")
    for step, batch in enumerate(tqdm(train_dataloader, total=num_iters, initial=start_iter)):
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device.type, dtype=precision):
            logits = compiled_model(x)
            loss = cross_entropy(logits, y)
            perplexity = torch.exp(loss)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.gradient_max_norm)
        lr = get_lr_cosine_schedule(step, args.max_lr, args.min_lr, args.warmup_iters, num_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if step % log_interval == 0:
            logger.info(f"Step {step}/{num_iters} loss: {loss.item():.4f} perplexity: {perplexity.item():.4f} lr: {lr:.6f}")
            if log_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train_loss": loss.item(),
                    "train_perplexity": perplexity.item(),
                    "learning_rate": lr,
                    "step": step
                })
        if step % checkpoint_interval == 0:
            logger.info(f"Saving checkpoint at step {step}")
            checkpoint_path = os.path.join(out, f"checkpoint_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path)
        if step > 0 and (step % args.validation_interval == 0 or step == num_iters - 1):
            logger.info(f"Running validation at step {step}")
            avg_loss, avg_perplexity = validate(model, val_dataloader, device, precision)
            logger.info(f"Validation at step {step} - loss: {avg_loss:.4f} perplexity: {avg_perplexity:.4f}")
            if avg_loss < best_loss:
                logger.info(f"New best validation loss: {avg_loss:.4f} (previous: {best_loss:.4f})")
                best_loss = avg_loss
                # Save best model with a different filename
                best_model_path = os.path.join(out, "best_model.pt")
                save_checkpoint(model, optimizer, step, best_model_path)
            logger.info("Generating sample text...")
            generated_text = run_generation(model, tokenizer, args.val_prompt, 
                           args.max_new_tokens, args.temperature, 
                           args.top_k, args.top_p, args.seed, device)
            logger.info(f"Generated text at step {step}:\n{generated_text}")
            if log_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "validation_loss": avg_loss,
                    "validation_perplexity": avg_perplexity,
                    "generated_text": generated_text
                })
            
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    if log_wandb and WANDB_AVAILABLE:
        wandb.log({
            "best_loss": best_loss,
            "num_iters": num_iters
        })
    logger.info(f"Best loss: {best_loss:.4f}")

    # Save final model
    final_model_path = os.path.join(out, "final_model.pt")
    save_checkpoint(model, optimizer, args.num_iters, final_model_path)

    # Log to Weights & Biases
    if args.log_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return 

def validate(
    model: TransformerLM,
    val_dataloader: IterableDataset,
    device: str,
    precision: torch.dtype
) -> float:
    logger.info("Starting validation...")
    model.eval()
    # Convert device string to torch.device object
    device = torch.device(device)
    with torch.inference_mode():
        total_loss = 0.0
        total_perplexity = 0.0
        total_tokens = 0
        num_batches = 0
        for batch in tqdm(val_dataloader, desc="Validating"):    
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(device.type, dtype=precision):
                logits = model(x)
                loss = cross_entropy(logits, y)
                perplexity = torch.exp(loss)
            total_loss += loss.item() * y.numel()
            total_perplexity += perplexity.item() * y.numel()
            total_tokens += y.numel()
            num_batches += 1
        avg_loss = total_loss / total_tokens
        avg_perplexity = total_perplexity / total_tokens
        logger.info(f"Validation complete - processed {num_batches} batches, {total_tokens} tokens")
        return avg_loss, avg_perplexity
        


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str]):
    tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
    logger.info(f"Tokenizer loaded from {vocab_path} and {merges_path} with vocabulary size {len(tokenizer.get_vocab())}")
    return tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM training with Transformer architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Dataset and tokenization args
    parser.add_argument("--train_dataset_path", type=str, default="../data/TS_train_encoded.npy",
                      help="Path to the training dataset (.txt file)")
    parser.add_argument("--val_dataset_path", type=str, default="../data/TS_valid_encoded.npy",
                      help="Path to the validation dataset (.txt file)")
    parser.add_argument("--vocab_size", type=int,
                      help="Size of the vocabulary", default=50257)
    parser.add_argument("--special_tokens", type=list[str],
                      help="List of special tokens to add to vocabulary", default=["<|endoftext|>"])
    parser.add_argument("--vocab_path", type=str, default = "bpe_10k_TSV2/tsv2-vocab.json",
                      help="Path to the vocabulary file")
    parser.add_argument("--merges_path", type=str, default = "bpe_10k_TSV2/tsv2-merges.txt",
                      help="Path to the merges file")
    
    # Model architecture args
    parser.add_argument("--d_model", type=int, default=512,
                      help="Dimension of the model embeddings and transformer layers")
    parser.add_argument("--num_layers", type=int, default=4,
                      help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16,
                      help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344,
                      help="Dimension of the feed-forward layer (typically 8/3 * d_model)")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                      help="RoPE theta parameter for positional embeddings")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=256,
                      help="Training batch size")
    parser.add_argument("--context_length", type=int, 
                      help="Maximum sequence length/context window", default=256)
    parser.add_argument("--num_iters", type=int,
                      help="Total number of training iterations", default=100000)
    parser.add_argument("--device", type=str, 
                      help="Device to train on (e.g. 'cuda:0' or 'cpu')", default="cuda:0")
    parser.add_argument("--out", type=str, default="checkpoints",
                      help="Directory to save model checkpoints")
    parser.add_argument("--validation_interval", type=int,
                      help="How often to validate the model", default=1000)
    parser.add_argument("--val-prompt", type=str, default=VAL_PROMPTS[0],help="Prompt to use for validation generation")
    
    # Optimizer args
    parser.add_argument("--max_lr", type=float, default=1e-3,
                      help="Maximum learning rate for cosine schedule")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                      help="Minimum learning rate for cosine schedule")
    parser.add_argument("--warmup_iters", type=int, default=10000,
                      help="Number of warmup iterations for learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay for AdamW optimizer")
    parser.add_argument("--gradient_max_norm", type=float, default=1.0,
                      help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--beta1", type=float, default=0.9,
                      help="Beta1 parameter for AdamW optimizer")
    parser.add_argument("--beta2", type=float, default=0.95,
                      help="Beta2 parameter for AdamW optimizer")
    
    # Logging and checkpointing args
    parser.add_argument("--log_wandb", type=bool, 
                      help="Whether to log to Weights & Biases", default=False)
    parser.add_argument("--log_interval", type=int, 
                      help="How often to log training metrics", default=1000)
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                      help="How often to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Temperature for generation")
    parser.add_argument("--top_k", type=int, default=10,
                      help="Top-k for generation")
    parser.add_argument("--top_p", type=float, default=0.0,
                      help="Top-p for generation")

    parser.add_argument("--max_new_tokens", type=int, default=100,
                      help="Maximum number of tokens to generate")
    parser.add_argument("--precision", type=str, default="float32",
                      help="Precision for training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                      help="Path to the checkpoint to resume from")

    args = parser.parse_args()
    
    # Set random seed
    set_all_seeds(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # check if use cuda
    if torch.cuda.is_available():
        args.device = "cuda:0"
        logger.info("CUDA is available, using GPU")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        args.device = "cpu"
        logger.info("CUDA is not available, using CPU")
    
    # Convert device string to torch.device object for model creation
    device = torch.device(args.device)
    
    # Load dataset from text file
    logger.info(f"Loading training dataset from {args.train_dataset_path}")
    train_dataset = np.load(args.train_dataset_path, mmap_mode="r")
    logger.info(f"Training dataset loaded, shape: {train_dataset.shape}")
    
    if args.val_dataset_path is not None:
        logger.info(f"Loading validation dataset from {args.val_dataset_path}")
        val_dataset = np.load(args.val_dataset_path, mmap_mode="r")
        logger.info(f"Validation dataset loaded, shape: {val_dataset.shape}")
    else:
        val_dataset = None
        logger.info("No validation dataset provided")
    
    logger.info(f"Loading tokenizer from {args.vocab_path} and {args.merges_path}")
    tokenizer = load_tokenizer(args.vocab_path, args.merges_path, args.special_tokens)
    logger.info(f"Tokenizer loaded with vocabulary size: {len(tokenizer.get_vocab())}")
    eot_id = tokenizer.token_to_id(args.special_tokens[0])

    train_dataloader = random_training_iterator(
        train_dataset, args.batch_size, args.context_length, device, args.num_iters
    )

    if val_dataset is not None:
        val_dataloader = SequentialValidationDataset(
            val_dataset, args.context_length, args.batch_size, device
        )
    else:
        val_dataloader = None

    # Load model
    logger.info("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.bfloat16
    )
    model.to(device)
    logger.info(f"Model initialized and moved to {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load optimizer
    logger.info("Initializing optimizer...")

    logger.info(f"Optimizer initialized with max_lr={args.max_lr}, weight_decay={args.weight_decay}")
    scheduler = get_lr_cosine_schedule(
        it = 0,
        max_learning_rate=args.max_lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.num_iters
    )
    

    train_LLM(
        model=model,
        tokenizer=tokenizer,
        num_iters=args.num_iters,
        device=args.device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        out=args.out,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        log_wandb=args.log_wandb,
        args=args
    )


    # Save model
    #save_checkpoint(model, optimizer, args.num_iters, args.out)

    # Log to Weights & Biases
    if args.log_wandb and WANDB_AVAILABLE:
        wandb.finish()

    




