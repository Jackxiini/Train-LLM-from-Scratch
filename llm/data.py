from typing import Iterator, Union, Tuple
import torch
import numpy as np
import numpy.typing as npt
from torch.utils.data import IterableDataset

def get_batch(dataset: npt.NDArray, 
              batch_size: int, 
              context_length: int, device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str | torch.device): PyTorch device string or torch.device object indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    start_idx = np.random.randint(0, len(dataset)-context_length, size = batch_size)
    x = np.array([dataset[i:i+context_length] for i in start_idx], dtype=np.int64)
    y = np.array([dataset[i+1:i+context_length+1] for i in start_idx], dtype=np.int64)

    # First create tensors on CPU
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    # Convert device to string for checking device type
    device_str = str(device)
    
    # Handle device placement and memory optimization
    if "cuda" in device_str:
        # Pin memory for faster CPU to GPU transfer
        x_tensor = x_tensor.pin_memory()
        y_tensor = y_tensor.pin_memory()
        # Move to GPU with non_blocking=True for async transfer
        x_tensor = x_tensor.to(device, non_blocking=True)
        y_tensor = y_tensor.to(device, non_blocking=True)
    elif "mps" in device_str:
        # For MPS (Apple Silicon), use channels_last memory format
        x_tensor = x_tensor.to(device, memory_format=torch.channels_last)
        y_tensor = y_tensor.to(device, memory_format=torch.channels_last)
    else:
        # For CPU, just move to device
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

    return x_tensor, y_tensor

def random_training_iterator(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: Union[str, torch.device],
    max_iter: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Yields `max_iter` random training batches using get_batch.
    """
    for _ in range(max_iter):
        yield get_batch(dataset, batch_size, context_length, device)


class SequentialValidationDataset(IterableDataset):
    def __init__(
        self,
        dataset: npt.NDArray,
        context_length: int,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        IterableDataset that yields sequential non-overlapping batches from a 1D tokenized dataset.

        Args:
            dataset: 1D numpy array of token IDs.
            context_length: Length of each sequence.
            batch_size: Number of sequences per batch.
            device: Device to move tensors to.
        """
        self.dataset = dataset
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

        self.total_tokens = len(dataset)
        self.num_sequences = (self.total_tokens - 1) // context_length

    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.
        """
        return (self.num_sequences - 1) // self.batch_size + 1

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Generate all sequences
        all_x = []
        all_y = []
        for i in range(self.num_sequences):
            start = i * self.context_length
            end = start + self.context_length + 1
            if end > self.total_tokens:
                break
            seq = self.dataset[start:end]
            all_x.append(seq[:-1])
            all_y.append(seq[1:])

        all_x = np.stack(all_x, dtype=np.int64)
        all_y = np.stack(all_y, dtype=np.int64)

        # Convert device to string for checking device type
        device_str = str(self.device)

        # Yield in batches
        for i in range(0, len(all_x), self.batch_size):
            # First create tensors on CPU
            xb = torch.from_numpy(all_x[i : i + self.batch_size])
            yb = torch.from_numpy(all_y[i : i + self.batch_size])

            if "cuda" in device_str:
                # Pin memory for faster CPU to GPU transfer
                xb = xb.pin_memory()
                yb = yb.pin_memory()
                # Move to GPU with non_blocking=True for async transfer
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
            else:
                # For CPU or other devices, just move to device
                xb = xb.to(self.device)
                yb = yb.to(self.device)

            yield xb, yb


