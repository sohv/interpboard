"""
Utility functions for the patching module.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from ..utils import safe_divide


def find_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Find a module by its name in the model."""
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise ValueError(f"Module {name} not found in model")


def get_activation_statistics(
    activations: torch.Tensor,
    dim: Optional[int] = None
) -> Dict[str, float]:
    """Compute statistics for activations."""
    stats = {}
    
    if dim is None:
        # Global statistics
        stats["mean"] = activations.mean().item()
        stats["std"] = activations.std().item()
        stats["min"] = activations.min().item()
        stats["max"] = activations.max().item()
        stats["norm_l2"] = torch.norm(activations, p=2).item()
        stats["norm_l1"] = torch.norm(activations, p=1).item()
    else:
        # Statistics along specific dimension
        stats["mean"] = activations.mean(dim=dim)
        stats["std"] = activations.std(dim=dim)
        stats["min"] = activations.min(dim=dim)[0]
        stats["max"] = activations.max(dim=dim)[0]
        
    return stats


def compute_activation_similarity(
    activation1: torch.Tensor,
    activation2: torch.Tensor,
    method: str = "cosine"
) -> float:
    """Compute similarity between two activations."""
    if method == "cosine":
        # Flatten activations
        act1_flat = activation1.view(-1)
        act2_flat = activation2.view(-1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(act1_flat.unsqueeze(0), act2_flat.unsqueeze(0))
        return similarity.item()
    
    elif method == "l2":
        # L2 distance (smaller is more similar)
        distance = torch.norm(activation1 - activation2, p=2)
        return -distance.item()  # Negative so higher is more similar
    
    elif method == "pearson":
        # Pearson correlation
        act1_flat = activation1.view(-1)
        act2_flat = activation2.view(-1)
        
        correlation = torch.corrcoef(torch.stack([act1_flat, act2_flat]))[0, 1]
        return correlation.item() if not torch.isnan(correlation) else 0.0
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def create_noise_injection_hook(
    noise_std: float = 0.1,
    positions: Optional[List[int]] = None
):
    """Create a hook that injects noise into activations."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            activation = output[0]
            rest = output[1:]
        else:
            activation = output
            rest = ()
        
        # Create noise
        noise = torch.randn_like(activation) * noise_std
        
        if positions is not None:
            # Only add noise at specific positions
            noise_mask = torch.zeros_like(activation)
            for pos in positions:
                if pos < activation.shape[1]:  # Assuming sequence dimension is 1
                    noise_mask[:, pos] = 1.0
            noise = noise * noise_mask
        
        noisy_activation = activation + noise
        
        return (noisy_activation,) + rest if rest else noisy_activation
    
    return hook


def create_zeroing_hook(
    positions: Optional[List[int]] = None,
    heads: Optional[List[int]] = None
):
    """Create a hook that zeros out specific components."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            activation = output[0]
            rest = output[1:]
        else:
            activation = output
            rest = ()
        
        modified_activation = activation.clone()
        
        if positions is not None:
            # Zero out specific positions
            for pos in positions:
                if pos < activation.shape[1]:
                    modified_activation[:, pos] = 0.0
        
        if heads is not None and len(activation.shape) >= 3:
            # Zero out specific heads (assuming head dimension exists)
            for head in heads:
                if head < activation.shape[2]:
                    modified_activation[:, :, head] = 0.0
        
        return (modified_activation,) + rest if rest else modified_activation
    
    return hook


def analyze_patch_impact_distribution(
    patch_results: List,
    metric: str = "logit_l2_diff"
) -> Dict[str, float]:
    """Analyze the distribution of patch impacts."""
    impacts = [result.metrics.get(metric, 0.0) for result in patch_results]
    impacts = np.array(impacts)
    
    return {
        "mean": np.mean(impacts),
        "std": np.std(impacts),
        "median": np.median(impacts),
        "q25": np.percentile(impacts, 25),
        "q75": np.percentile(impacts, 75),
        "min": np.min(impacts),
        "max": np.max(impacts),
        "range": np.max(impacts) - np.min(impacts)
    }


def rank_components_by_impact(
    patch_results: List,
    metric: str = "logit_l2_diff",
    top_k: Optional[int] = None
) -> List[Tuple]:
    """Rank components by their patch impact."""
    component_impacts = []
    
    for result in patch_results:
        impact = result.metrics.get(metric, 0.0)
        location = result.patch_location
        
        # Create component identifier
        component_id = f"L{location.layer_idx}_{location.component}"
        if location.head_idx is not None:
            component_id += f"_H{location.head_idx}"
        
        component_impacts.append((component_id, impact, result))
    
    # Sort by impact (descending)
    component_impacts.sort(key=lambda x: x[1], reverse=True)
    
    if top_k is not None:
        component_impacts = component_impacts[:top_k]
    
    return component_impacts


def compute_intervention_strength(
    original_activation: torch.Tensor,
    patched_activation: torch.Tensor
) -> float:
    """Compute the strength of an intervention."""
    diff = original_activation - patched_activation
    
    # Normalize by original activation magnitude
    original_norm = torch.norm(original_activation, p=2)
    diff_norm = torch.norm(diff, p=2)
    
    if original_norm > 1e-8:
        strength = (diff_norm / original_norm).item()
    else:
        strength = diff_norm.item()
    
    return strength


def create_gradient_patching_hook(
    target_gradient: torch.Tensor,
    strength: float = 1.0
):
    """Create a hook that modifies gradients during backpropagation."""
    def hook(grad):
        if grad is None:
            return None
        
        # Apply target gradient with specified strength
        modified_grad = grad + strength * target_gradient.to(grad.device)
        return modified_grad
    
    return hook


def find_optimal_patch_strength(
    patcher,
    text: str,
    patch_location,
    patch_activation: torch.Tensor,
    target_metric: str = "logit_l2_diff",
    strengths: Optional[List[float]] = None
) -> Tuple[float, float]:
    """Find optimal strength for a patch to maximize target metric."""
    if strengths is None:
        strengths = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    best_strength = 0.0
    best_score = 0.0
    
    for strength in strengths:
        scaled_patch = patch_activation * strength
        
        result = patcher.patch_activation(
            text,
            patch_location,
            patch_type="custom",
            custom_value=scaled_patch
        )
        
        score = result.metrics.get(target_metric, 0.0)
        
        if score > best_score:
            best_score = score
            best_strength = strength
    
    return best_strength, best_score


def compute_patch_consistency(
    patch_results: List,
    metric: str = "logit_l2_diff"
) -> float:
    """Compute consistency of patch effects across multiple runs."""
    impacts = [result.metrics.get(metric, 0.0) for result in patch_results]
    
    if len(impacts) < 2:
        return 1.0
    
    # Compute coefficient of variation
    mean_impact = np.mean(impacts)
    std_impact = np.std(impacts)
    
    if mean_impact > 1e-8:
        consistency = 1.0 - (std_impact / mean_impact)
    else:
        consistency = 1.0 if std_impact < 1e-8 else 0.0
    
    return max(0.0, consistency)


def detect_redundant_components(
    patch_results: List,
    similarity_threshold: float = 0.9,
    metric: str = "logit_l2_diff"
) -> List[List[str]]:
    """Detect groups of components with redundant effects."""
    # Create impact vectors for each component
    component_impacts = {}
    
    for result in patch_results:
        location = result.patch_location
        component_id = f"L{location.layer_idx}_{location.component}"
        if location.head_idx is not None:
            component_id += f"_H{location.head_idx}"
        
        impact = result.metrics.get(metric, 0.0)
        component_impacts[component_id] = impact
    
    # Find similar components
    components = list(component_impacts.keys())
    redundant_groups = []
    processed = set()
    
    for i, comp1 in enumerate(components):
        if comp1 in processed:
            continue
        
        group = [comp1]
        impact1 = component_impacts[comp1]
        
        for j, comp2 in enumerate(components[i+1:], i+1):
            if comp2 in processed:
                continue
            
            impact2 = component_impacts[comp2]
            
            # Simple similarity based on impact magnitude
            if abs(impact1) > 1e-8 and abs(impact2) > 1e-8:
                similarity = min(impact1, impact2) / max(impact1, impact2)
                if similarity >= similarity_threshold:
                    group.append(comp2)
                    processed.add(comp2)
        
        if len(group) > 1:
            redundant_groups.append(group)
        
        processed.add(comp1)
    
    return redundant_groups