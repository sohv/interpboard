"""
Activation patching utilities for mechanistic interpretability.

This module provides tools for patching activations in transformer models
to understand causal relationships between components and outputs.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils import tokenize_input, get_model_info, safe_divide

logger = logging.getLogger(__name__)


class PatchType(Enum):
    """Types of activation patches."""
    ZERO = "zero"
    MEAN = "mean" 
    RANDOM = "random"
    NOISE = "noise"
    CUSTOM = "custom"


@dataclass
class PatchLocation:
    """Specification for where to apply a patch."""
    layer_idx: int
    component: str  # "attention", "mlp", "residual"
    head_idx: Optional[int] = None  # For attention heads
    position_idx: Optional[Union[int, List[int]]] = None  # Token positions


@dataclass
class PatchResult:
    """Results from an activation patching experiment."""
    original_output: torch.Tensor
    patched_output: torch.Tensor
    patch_location: PatchLocation
    patch_type: PatchType
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


class ActivationPatcher:
    """Main class for activation patching experiments."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model_info = get_model_info(model)
        
        # Store original activations for mean/baseline computation
        self._activation_cache = {}
        self._hooks = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_hooks()
    
    def cleanup_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def get_component_module(self, layer_idx: int, component: str) -> torch.nn.Module:
        """Get the module for a specific component."""
        # This needs to be adapted based on model architecture
        if "gpt2" in self.model.config.model_type.lower():
            base_layer = self.model.transformer.h[layer_idx]
            if component == "attention":
                return base_layer.attn
            elif component == "mlp":
                return base_layer.mlp
            elif component == "residual":
                return base_layer
        elif "llama" in self.model.config.model_type.lower():
            base_layer = self.model.model.layers[layer_idx]
            if component == "attention":
                return base_layer.self_attn
            elif component == "mlp":
                return base_layer.mlp
            elif component == "residual":
                return base_layer
        else:
            raise NotImplementedError(f"Model type {self.model.config.model_type} not supported")
    
    def compute_baseline_activations(
        self,
        text_samples: List[str],
        patch_locations: List[PatchLocation],
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """Compute baseline activations for mean patching."""
        logger.info(f"Computing baseline activations from {len(text_samples)} samples")
        
        baselines = {}
        activation_collector = {}
        
        def make_collector_hook(location_key: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output
                
                if location_key not in activation_collector:
                    activation_collector[location_key] = []
                activation_collector[location_key].append(activation.detach().cpu())
            return hook
        
        # Register hooks for each patch location
        hooks = []
        for location in patch_locations:
            module = self.get_component_module(location.layer_idx, location.component)
            location_key = f"{location.layer_idx}_{location.component}"
            if location.head_idx is not None:
                location_key += f"_{location.head_idx}"
            
            hook = module.register_forward_hook(make_collector_hook(location_key))
            hooks.append(hook)
        
        try:
            # Collect activations from samples
            sample_count = 0
            for text in text_samples[:num_samples]:
                inputs = tokenize_input(text, self.tokenizer)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
            
            # Compute means
            for location_key, activations in activation_collector.items():
                stacked = torch.stack(activations, dim=0)
                baselines[location_key] = stacked.mean(dim=0)
                
        finally:
            for hook in hooks:
                hook.remove()
        
        logger.info(f"Computed baselines for {len(baselines)} components")
        return baselines
    
    def create_patch_value(
        self,
        original_activation: torch.Tensor,
        patch_type: PatchType,
        baseline: Optional[torch.Tensor] = None,
        noise_std: float = 0.1,
        custom_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Create the patching value based on patch type."""
        if patch_type == PatchType.ZERO:
            return torch.zeros_like(original_activation)
        
        elif patch_type == PatchType.MEAN:
            if baseline is None:
                return torch.zeros_like(original_activation)
            return baseline.to(original_activation.device)
        
        elif patch_type == PatchType.RANDOM:
            return torch.randn_like(original_activation)
        
        elif patch_type == PatchType.NOISE:
            noise = torch.randn_like(original_activation) * noise_std
            return original_activation + noise
        
        elif patch_type == PatchType.CUSTOM:
            if custom_value is None:
                raise ValueError("Custom patch value must be provided for CUSTOM patch type")
            return custom_value.to(original_activation.device)
        
        else:
            raise ValueError(f"Unknown patch type: {patch_type}")
    
    def apply_attention_head_patch(
        self,
        attention_output: torch.Tensor,
        head_idx: int,
        patch_value: torch.Tensor,
        position_idx: Optional[Union[int, List[int]]] = None
    ) -> torch.Tensor:
        """Apply patch to specific attention head."""
        # attention_output shape: [batch, seq_len, num_heads, head_dim]
        batch_size, seq_len, num_heads, head_dim = attention_output.shape
        
        if head_idx >= num_heads:
            raise ValueError(f"Head index {head_idx} >= num_heads {num_heads}")
        
        patched_output = attention_output.clone()
        
        if position_idx is None:
            # Patch entire head
            patched_output[:, :, head_idx, :] = patch_value
        else:
            # Patch specific positions
            if isinstance(position_idx, int):
                position_idx = [position_idx]
            for pos in position_idx:
                if pos < seq_len:
                    patched_output[:, pos, head_idx, :] = patch_value[:, pos] if patch_value.ndim > 1 else patch_value
        
        return patched_output
    
    def patch_activation(
        self,
        text: str,
        patch_location: PatchLocation,
        patch_type: PatchType = PatchType.ZERO,
        baseline: Optional[torch.Tensor] = None,
        custom_value: Optional[torch.Tensor] = None,
        noise_std: float = 0.1,
        return_activations: bool = False
    ) -> PatchResult:
        """
        Patch activations at a specific location and return the results.
        
        Args:
            text: Input text to process
            patch_location: Where to apply the patch
            patch_type: Type of patch to apply
            baseline: Baseline activation for mean patching
            custom_value: Custom value for custom patching
            noise_std: Standard deviation for noise patching
            return_activations: Whether to return intermediate activations
            
        Returns:
            PatchResult containing original and patched outputs
        """
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get original output
        with torch.no_grad():
            original_output = self.model(**inputs)
        
        # Set up patching hook
        patched_output = None
        activations = {} if return_activations else None
        
        def make_patch_hook():
            def hook(module, input, output):
                nonlocal patched_output
                
                if isinstance(output, tuple):
                    activation = output[0]
                    rest = output[1:]
                else:
                    activation = output
                    rest = ()
                
                # Store original if needed
                if return_activations:
                    activations[f"{patch_location.layer_idx}_{patch_location.component}_original"] = activation.detach()
                
                # Create patch value
                patch_value = self.create_patch_value(
                    activation,
                    patch_type,
                    baseline,
                    noise_std,
                    custom_value
                )
                
                # Apply patch based on component type
                if patch_location.component == "attention" and patch_location.head_idx is not None:
                    patched_activation = self.apply_attention_head_patch(
                        activation,
                        patch_location.head_idx,
                        patch_value,
                        patch_location.position_idx
                    )
                else:
                    # Full component patch
                    if patch_location.position_idx is None:
                        patched_activation = patch_value
                    else:
                        patched_activation = activation.clone()
                        if isinstance(patch_location.position_idx, int):
                            patched_activation[:, patch_location.position_idx] = patch_value[:, patch_location.position_idx]
                        else:
                            for pos in patch_location.position_idx:
                                patched_activation[:, pos] = patch_value[:, pos]
                
                # Store patched activation if needed
                if return_activations:
                    activations[f"{patch_location.layer_idx}_{patch_location.component}_patched"] = patched_activation.detach()
                
                return (patched_activation,) + rest if rest else patched_activation
            
            return hook
        
        # Register hook and run patched forward pass
        module = self.get_component_module(patch_location.layer_idx, patch_location.component)
        hook = module.register_forward_hook(make_patch_hook())
        
        try:
            with torch.no_grad():
                patched_output = self.model(**inputs)
        finally:
            hook.remove()
        
        # Compute metrics
        metrics = self.compute_patch_metrics(original_output, patched_output, inputs)
        
        # Create result
        result = PatchResult(
            original_output=original_output,
            patched_output=patched_output,
            patch_location=patch_location,
            patch_type=patch_type,
            metrics=metrics,
            metadata={
                "text": text,
                "activations": activations,
                "patch_params": {
                    "noise_std": noise_std,
                    "baseline_used": baseline is not None,
                    "custom_value_used": custom_value is not None
                }
            }
        )
        
        return result
    
    def compute_patch_metrics(
        self,
        original_output,
        patched_output,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics comparing original and patched outputs."""
        metrics = {}
        
        # Get logits
        if hasattr(original_output, "logits"):
            orig_logits = original_output.logits
            patch_logits = patched_output.logits
        else:
            orig_logits = original_output
            patch_logits = patched_output
        
        # Logit difference (L2 norm)
        logit_diff = torch.norm(orig_logits - patch_logits, p=2, dim=-1).mean()
        metrics["logit_l2_diff"] = logit_diff.item()
        
        # Probability difference (KL divergence)
        orig_probs = F.softmax(orig_logits, dim=-1)
        patch_probs = F.softmax(patch_logits, dim=-1)
        kl_div = F.kl_div(patch_probs.log(), orig_probs, reduction="batchmean")
        metrics["kl_divergence"] = kl_div.item()
        
        # Top-k rank changes
        for k in [1, 5, 10]:
            orig_topk = torch.topk(orig_logits, k, dim=-1).indices
            patch_topk = torch.topk(patch_logits, k, dim=-1).indices
            
            # Compute overlap
            overlap = 0
            for i in range(orig_topk.shape[0]):  # batch
                for j in range(orig_topk.shape[1]):  # sequence
                    orig_set = set(orig_topk[i, j].cpu().numpy())
                    patch_set = set(patch_topk[i, j].cpu().numpy())
                    overlap += len(orig_set.intersection(patch_set)) / k
            
            metrics[f"top_{k}_overlap"] = overlap / (orig_topk.shape[0] * orig_topk.shape[1])
        
        # Prediction change for next token
        if orig_logits.shape[1] > 1:  # Multi-token sequence
            last_orig = orig_logits[:, -1, :]  # Last token logits
            last_patch = patch_logits[:, -1, :]
            
            orig_pred = torch.argmax(last_orig, dim=-1)
            patch_pred = torch.argmax(last_patch, dim=-1)
            
            prediction_changed = (orig_pred != patch_pred).float().mean()
            metrics["prediction_changed"] = prediction_changed.item()
        
        return metrics
    
    def run_systematic_ablation(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        components: Optional[List[str]] = None,
        heads: Optional[List[int]] = None,
        patch_type: PatchType = PatchType.ZERO,
        baseline_texts: Optional[List[str]] = None
    ) -> List[PatchResult]:
        """
        Run systematic ablation across multiple components.
        
        Args:
            text: Input text to analyze
            layers: Layer indices to test (default: all)
            components: Components to test (default: ["attention", "mlp"])
            heads: Attention heads to test (default: all)
            patch_type: Type of patch to apply
            baseline_texts: Texts for computing baselines (for mean patching)
            
        Returns:
            List of PatchResult objects
        """
        if layers is None:
            layers = list(range(self.model_info["num_layers"]))
        
        if components is None:
            components = ["attention", "mlp"]
        
        if heads is None and "attention" in components:
            heads = list(range(self.model_info["num_heads"]))
        
        # Compute baselines if needed
        baselines = {}
        if patch_type == PatchType.MEAN and baseline_texts:
            patch_locations = []
            for layer in layers:
                for component in components:
                    if component == "attention" and heads:
                        for head in heads:
                            patch_locations.append(PatchLocation(layer, component, head))
                    else:
                        patch_locations.append(PatchLocation(layer, component))
            
            baselines = self.compute_baseline_activations(baseline_texts, patch_locations)
        
        # Run ablations
        results = []
        total_experiments = len(layers) * len(components)
        if "attention" in components and heads:
            total_experiments += len(layers) * len(heads)
        
        logger.info(f"Running {total_experiments} ablation experiments")
        
        for layer in layers:
            for component in components:
                if component == "attention" and heads:
                    # Test individual attention heads
                    for head in heads:
                        location = PatchLocation(layer, component, head)
                        baseline_key = f"{layer}_{component}_{head}"
                        baseline = baselines.get(baseline_key)
                        
                        result = self.patch_activation(
                            text, location, patch_type, baseline
                        )
                        results.append(result)
                else:
                    # Test full component
                    location = PatchLocation(layer, component)
                    baseline_key = f"{layer}_{component}"
                    baseline = baselines.get(baseline_key)
                    
                    result = self.patch_activation(
                        text, location, patch_type, baseline
                    )
                    results.append(result)
        
        logger.info(f"Completed {len(results)} ablation experiments")
        return results
    
    def find_critical_components(
        self,
        text: str,
        metric: str = "logit_l2_diff",
        threshold: float = 0.1,
        **ablation_kwargs
    ) -> List[Tuple[PatchLocation, float]]:
        """
        Find components that are critical for the model's output.
        
        Args:
            text: Input text to analyze
            metric: Metric to use for ranking criticality
            threshold: Minimum metric value to consider critical
            **ablation_kwargs: Additional arguments for systematic ablation
            
        Returns:
            List of (PatchLocation, metric_value) tuples sorted by criticality
        """
        results = self.run_systematic_ablation(text, **ablation_kwargs)
        
        critical_components = []
        for result in results:
            metric_value = result.metrics.get(metric, 0.0)
            if metric_value >= threshold:
                critical_components.append((result.patch_location, metric_value))
        
        # Sort by metric value (descending)
        critical_components.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(critical_components)} critical components above threshold {threshold}")
        return critical_components