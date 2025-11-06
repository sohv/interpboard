"""
Core utility functions for the LLM interpretability library.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **model_kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from HuggingFace.
    
    Args:
        model_name: Name or path of the model
        device: Device to load model on ('cuda', 'cpu', 'auto')
        torch_dtype: Torch dtype for model weights
        **model_kwargs: Additional arguments for model loading
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    logger.info(f"Loading model {model_name} on {device}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "auto" else "auto",
        output_attentions=True,
        **model_kwargs
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Successfully loaded {model_name}")
    return model, tokenizer


def get_model_info(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Extract key information about the model architecture.
    
    Args:
        model: The transformer model
        
    Returns:
        Dictionary with model information
    """
    config = model.config
    
    info = {
        "model_type": config.model_type,
        "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", None)),
        "num_heads": getattr(config, "num_attention_heads", getattr(config, "n_head", None)),
        "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", None)),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", getattr(config, "n_positions", None)),
    }
    
    return {k: v for k, v in info.items() if v is not None}


def tokenize_input(
    text: Union[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    return_tensors: str = "pt",
    **tokenize_kwargs
) -> Dict[str, torch.Tensor]:
    """
    Tokenize input text with proper handling.
    
    Args:
        text: Input text or list of texts
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        return_tensors: Format for returned tensors
        **tokenize_kwargs: Additional tokenization arguments
    
    Returns:
        Tokenized inputs as tensors
    """
    if max_length is None:
        max_length = getattr(tokenizer, "model_max_length", 512)
        max_length = min(max_length, 512)  # Cap at reasonable length
    
    inputs = tokenizer(
        text,
        return_tensors=return_tensors,
        max_length=max_length,
        truncation=True,
        padding=True,
        **tokenize_kwargs
    )
    
    return inputs


def get_layer_names(model: PreTrainedModel) -> List[str]:
    """
    Get the names of all layers in the model.
    
    Args:
        model: The transformer model
        
    Returns:
        List of layer names
    """
    layer_names = []
    for name, _ in model.named_modules():
        if any(layer_type in name for layer_type in ["layer", "block", "h."]):
            layer_names.append(name)
    
    return layer_names


def get_attention_heads_info(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Get information about attention heads in the model.
    
    Args:
        model: The transformer model
        
    Returns:
        Dictionary with attention head information
    """
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
    num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 12))
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 768))
    head_dim = hidden_size // num_heads
    
    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "total_heads": num_layers * num_heads
    }


def extract_activations(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
    layer_names: Optional[List[str]] = None,
    output_attentions: bool = True,
    output_hidden_states: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extract activations from specific layers during a forward pass.
    
    Args:
        model: The transformer model
        inputs: Tokenized inputs
        layer_names: Specific layers to extract from
        output_attentions: Whether to output attention weights
        output_hidden_states: Whether to output hidden states
        
    Returns:
        Dictionary of extracted activations
    """
    activations = {}
    
    def make_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    if layer_names:
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
    
    try:
        # Forward pass
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
        
        # Add standard outputs
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            for i, hidden_state in enumerate(outputs.hidden_states):
                activations[f"hidden_state_{i}"] = hidden_state
                
        if hasattr(outputs, "attentions") and outputs.attentions:
            for i, attention in enumerate(outputs.attentions):
                activations[f"attention_{i}"] = attention
                
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    return activations


def compute_attention_rollout(
    attention_weights: torch.Tensor,
    discard_ratio: float = 0.9
) -> torch.Tensor:
    """
    Compute attention rollout following Abnar & Zuidema (2020).
    
    Args:
        attention_weights: Attention weights [batch, layers, heads, seq_len, seq_len]
        discard_ratio: Ratio of attention to discard in each step
        
    Returns:
        Rolled out attention weights
    """
    batch_size, num_layers, num_heads, seq_len, _ = attention_weights.shape
    
    # Average across heads
    attention_weights = attention_weights.mean(dim=2)  # [batch, layers, seq_len, seq_len]
    
    # Add residual connections
    residual_att = torch.eye(seq_len, device=attention_weights.device)
    residual_att = residual_att.expand(batch_size, num_layers, seq_len, seq_len)
    attention_weights = attention_weights + residual_att
    
    # Normalize
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    # Rollout
    joint_attentions = torch.zeros_like(attention_weights[:, 0])  # [batch, seq_len, seq_len]
    joint_attentions[:] = attention_weights[:, 0]
    
    for i in range(1, num_layers):
        joint_attentions = torch.matmul(attention_weights[:, i], joint_attentions)
    
    return joint_attentions


def create_token_mask(
    tokens: torch.Tensor,
    target_token_ids: Union[int, List[int]],
    tokenizer: PreTrainedTokenizer
) -> torch.Tensor:
    """
    Create a mask for specific tokens.
    
    Args:
        tokens: Token tensor [batch_size, seq_len]
        target_token_ids: Token ID(s) to mask
        tokenizer: Tokenizer for special tokens
        
    Returns:
        Boolean mask tensor
    """
    if isinstance(target_token_ids, int):
        target_token_ids = [target_token_ids]
    
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in target_token_ids:
        mask |= (tokens == token_id)
    
    # Exclude special tokens
    special_tokens = [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
    special_tokens = [t for t in special_tokens if t is not None]
    
    for special_token_id in special_tokens:
        mask &= (tokens != special_token_id)
    
    return mask


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Safely divide tensors, avoiding division by zero.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        epsilon: Small value to add to denominator
        
    Returns:
        Division result
    """
    return numerator / (denominator + epsilon)


def normalize_tensor(
    tensor: torch.Tensor,
    dim: int = -1,
    method: str = "l2"
) -> torch.Tensor:
    """
    Normalize a tensor along a given dimension.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize along
        method: Normalization method ('l2', 'l1', 'max')
        
    Returns:
        Normalized tensor
    """
    if method == "l2":
        norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    elif method == "l1":
        norm = torch.norm(tensor, p=1, dim=dim, keepdim=True)
    elif method == "max":
        norm = torch.max(torch.abs(tensor), dim=dim, keepdim=True)[0]
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return safe_divide(tensor, norm)


def batch_process(
    data: List[Any],
    process_fn: callable,
    batch_size: int = 8,
    show_progress: bool = True
) -> List[Any]:
    """
    Process data in batches.
    
    Args:
        data: List of data items
        process_fn: Function to process each batch
        batch_size: Size of each batch
        show_progress: Whether to show progress bar
        
    Returns:
        List of processed results
    """
    if show_progress:
        try:
            from tqdm import tqdm
            data = tqdm(data, desc="Processing")
        except ImportError:
            pass  # Continue without progress bar if tqdm not available
    
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_results = process_fn(batch)
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
    
    return results


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = "pickle"
) -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
        format: Save format ('pickle', 'json', 'npz')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pickle":
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
    elif format == "json":
        import json
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
    elif format == "npz":
        np.savez_compressed(filepath, **results)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        filepath: Path to load file
        
    Returns:
        Loaded results dictionary
    """
    filepath = Path(filepath)
    
    if filepath.suffix == ".pkl":
        import pickle
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif filepath.suffix == ".json":
        import json
        with open(filepath, "r") as f:
            return json.load(f)
    elif filepath.suffix == ".npz":
        data = np.load(filepath)
        return {key: data[key] for key in data.keys()}
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")