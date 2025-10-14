"""
Causal tracing implementation for understanding causal pathways in transformer models.

This module implements causal tracing as described in "Locating and Editing Factual 
Associations in GPT" and related work, allowing us to trace which components are 
causally responsible for specific model behaviors.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from .activ_patch import ActivationPatcher, PatchLocation, PatchType, PatchResult
from ..utils import tokenize_input

logger = logging.getLogger(__name__)


@dataclass
class CausalTraceConfig:
    """Configuration for causal tracing experiments."""
    noise_level: float = 0.1
    n_samples: int = 10
    layers_to_trace: Optional[List[int]] = None
    components_to_trace: List[str] = None
    trace_heads: bool = True
    
    def __post_init__(self):
        if self.components_to_trace is None:
            self.components_to_trace = ["attention", "mlp"]


@dataclass 
class CausalTraceResult:
    """Results from causal tracing experiment."""
    subject_token_positions: List[int]
    attribute_token_position: int
    trace_results: Dict[str, Dict[str, float]]  # {layer_component: {metric: value}}
    baseline_score: float
    corrupted_score: float
    metadata: Dict


class CausalTracer:
    """Implements causal tracing for transformer models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.patcher = ActivationPatcher(model, tokenizer, device)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.cleanup_hooks()
    
    def corrupt_input(
        self,
        text: str,
        subject_positions: List[int],
        noise_level: float = 0.1,
        n_samples: int = 10
    ) -> Tuple[str, torch.Tensor]:
        """
        Create corrupted version of input by adding noise to subject tokens.
        
        Args:
            text: Original input text
            subject_positions: Token positions to corrupt
            noise_level: Amount of noise to add
            n_samples: Number of noise samples to average over
            
        Returns:
            Tuple of (corrupted_text, clean_baseline_activations)
        """
        # Tokenize original text
        inputs = tokenize_input(text, self.tokenizer)
        input_ids = inputs["input_ids"]
        
        # Get baseline activations from clean input
        with torch.no_grad():
            clean_output = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        
        # Create corrupted versions by replacing subject tokens
        corrupted_texts = []
        for _ in range(n_samples):
            corrupted_ids = input_ids.clone()
            
            for pos in subject_positions:
                if pos < corrupted_ids.shape[1]:
                    # Replace with random token from vocabulary
                    random_token = torch.randint(
                        0, self.tokenizer.vocab_size, (1,), device=corrupted_ids.device
                    )
                    corrupted_ids[0, pos] = random_token
            
            corrupted_text = self.tokenizer.decode(corrupted_ids[0], skip_special_tokens=True)
            corrupted_texts.append(corrupted_text)
        
        # Use the first corrupted text as the primary corruption
        primary_corrupted = corrupted_texts[0]
        
        return primary_corrupted, clean_output
    
    def compute_restoration_effect(
        self,
        clean_text: str,
        corrupted_text: str,
        patch_location: PatchLocation,
        target_position: int,
        metric: str = "logit_diff"
    ) -> float:
        """
        Compute how much restoring clean activations at a location recovers performance.
        
        Args:
            clean_text: Original clean text
            corrupted_text: Corrupted version of text
            patch_location: Where to restore clean activations
            target_position: Token position to measure effect on
            metric: Metric to use for measuring effect
            
        Returns:
            Restoration effect score
        """
        # Get outputs for clean, corrupted, and restored versions
        clean_inputs = tokenize_input(clean_text, self.tokenizer)
        corrupted_inputs = tokenize_input(corrupted_text, self.tokenizer)
        
        # Move to device
        clean_inputs = {k: v.to(self.device) for k, v in clean_inputs.items()}
        corrupted_inputs = {k: v.to(self.device) for k, v in corrupted_inputs.items()}
        
        # Get clean and corrupted outputs
        with torch.no_grad():
            clean_output = self.model(**clean_inputs)
            corrupted_output = self.model(**corrupted_inputs)
        
        # Store clean activations for patching
        clean_activations = {}
        
        def store_clean_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    clean_activations[name] = output[0].detach()
                else:
                    clean_activations[name] = output.detach()
            return hook
        
        # Register hook to capture clean activations
        module = self.patcher.get_component_module(patch_location.layer_idx, patch_location.component)
        location_name = f"{patch_location.layer_idx}_{patch_location.component}"
        if patch_location.head_idx is not None:
            location_name += f"_{patch_location.head_idx}"
        
        hook = module.register_forward_hook(store_clean_activation(location_name))
        
        try:
            # Forward pass to get clean activations
            with torch.no_grad():
                _ = self.model(**clean_inputs)
            
            # Now patch corrupted input with clean activations
            result = self.patcher.patch_activation(
                corrupted_text,
                patch_location,
                PatchType.CUSTOM,
                custom_value=clean_activations[location_name]
            )
            restored_output = result.patched_output
            
        finally:
            hook.remove()
        
        # Compute metric scores
        clean_score = self._compute_target_score(clean_output, target_position, metric)
        corrupted_score = self._compute_target_score(corrupted_output, target_position, metric)
        restored_score = self._compute_target_score(restored_output, target_position, metric)
        
        # Compute restoration effect (how much of the damage was repaired)
        if abs(clean_score - corrupted_score) < 1e-8:
            return 0.0
        
        restoration_effect = (restored_score - corrupted_score) / (clean_score - corrupted_score)
        return restoration_effect
    
    def _compute_target_score(
        self,
        output,
        target_position: int,
        metric: str = "logit_diff"
    ) -> float:
        """Compute score for target position using specified metric."""
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output
        
        if metric == "logit_diff":
            # Difference between top 2 logits
            if target_position >= logits.shape[1]:
                return 0.0
            
            target_logits = logits[0, target_position, :]
            top2 = torch.topk(target_logits, 2)
            return (top2.values[0] - top2.values[1]).item()
        
        elif metric == "prob_target":
            # Probability of the target token (assumes we know target)
            target_logits = logits[0, target_position, :]
            probs = F.softmax(target_logits, dim=-1)
            return torch.max(probs).item()
        
        elif metric == "entropy":
            # Negative entropy (higher is more confident)
            target_logits = logits[0, target_position, :]
            probs = F.softmax(target_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            return -entropy.item()
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def trace_causal_effect(
        self,
        text: str,
        subject_tokens: List[str],
        target_token_position: int,
        config: Optional[CausalTraceConfig] = None
    ) -> CausalTraceResult:
        """
        Perform causal tracing to identify which components are responsible for a behavior.
        
        Args:
            text: Input text containing subject and target
            subject_tokens: List of subject tokens to corrupt
            target_token_position: Position of token to measure effect on
            config: Configuration for tracing
            
        Returns:
            CausalTraceResult with tracing results
        """
        if config is None:
            config = CausalTraceConfig()
        
        # Find subject token positions
        inputs = tokenize_input(text, self.tokenizer)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        subject_positions = []
        for subject_token in subject_tokens:
            subject_token_ids = self.tokenizer.encode(subject_token, add_special_tokens=False)
            for i, token_id in enumerate(inputs["input_ids"][0]):
                if token_id.item() in subject_token_ids:
                    subject_positions.append(i)
        
        if not subject_positions:
            logger.warning("No subject tokens found in input text")
            return CausalTraceResult([], target_token_position, {}, 0.0, 0.0, {})
        
        logger.info(f"Found subject tokens at positions: {subject_positions}")
        
        # Create corrupted input
        corrupted_text, clean_output = self.corrupt_input(
            text, subject_positions, config.noise_level, config.n_samples
        )
        
        # Compute baseline and corrupted scores
        baseline_score = self._compute_target_score(clean_output, target_token_position)
        
        corrupted_inputs = tokenize_input(corrupted_text, self.tokenizer)
        corrupted_inputs = {k: v.to(self.device) for k, v in corrupted_inputs.items()}
        
        with torch.no_grad():
            corrupted_output = self.model(**corrupted_inputs)
        corrupted_score = self._compute_target_score(corrupted_output, target_token_position)
        
        # Determine layers to trace
        num_layers = self.patcher.model_info["num_layers"]
        if config.layers_to_trace is None:
            layers_to_trace = list(range(num_layers))
        else:
            layers_to_trace = config.layers_to_trace
        
        # Trace causal effects
        trace_results = {}
        
        for layer_idx in layers_to_trace:
            for component in config.components_to_trace:
                
                if component == "attention" and config.trace_heads:
                    # Trace individual attention heads
                    num_heads = self.patcher.model_info["num_heads"]
                    for head_idx in range(num_heads):
                        location = PatchLocation(layer_idx, component, head_idx)
                        
                        effect = self.compute_restoration_effect(
                            text, corrupted_text, location, target_token_position
                        )
                        
                        key = f"L{layer_idx}H{head_idx}"
                        trace_results[key] = {"restoration_effect": effect}
                        
                else:
                    # Trace full component
                    location = PatchLocation(layer_idx, component)
                    
                    effect = self.compute_restoration_effect(
                        text, corrupted_text, location, target_token_position
                    )
                    
                    key = f"L{layer_idx}_{component}"
                    trace_results[key] = {"restoration_effect": effect}
        
        result = CausalTraceResult(
            subject_token_positions=subject_positions,
            attribute_token_position=target_token_position,
            trace_results=trace_results,
            baseline_score=baseline_score,
            corrupted_score=corrupted_score,
            metadata={
                "original_text": text,
                "corrupted_text": corrupted_text,
                "subject_tokens": subject_tokens,
                "config": config
            }
        )
        
        return result
    
    def analyze_information_flow(
        self,
        text: str,
        subject_tokens: List[str],
        target_token_position: int,
        threshold: float = 0.1
    ) -> Dict[str, List[str]]:
        """
        Analyze information flow by identifying early, middle, and late components.
        
        Args:
            text: Input text
            subject_tokens: Subject tokens to trace
            target_token_position: Target position
            threshold: Minimum effect to consider significant
            
        Returns:
            Dictionary categorizing components by their role in information flow
        """
        result = self.trace_causal_effect(text, subject_tokens, target_token_position)
        
        # Categorize components by layer depth
        early_components = []  # First third of layers
        middle_components = []  # Middle third
        late_components = []  # Last third
        
        num_layers = self.patcher.model_info["num_layers"]
        early_cutoff = num_layers // 3
        late_cutoff = 2 * num_layers // 3
        
        for component_name, metrics in result.trace_results.items():
            effect = metrics.get("restoration_effect", 0.0)
            
            if abs(effect) >= threshold:
                # Extract layer number
                if component_name.startswith("L"):
                    layer_num = int(component_name.split("H")[0][1:])  # Handle LxHy format
                    if "H" not in component_name:  # Handle Lx_component format
                        layer_num = int(component_name.split("_")[0][1:])
                    
                    if layer_num < early_cutoff:
                        early_components.append(component_name)
                    elif layer_num < late_cutoff:
                        middle_components.append(component_name)
                    else:
                        late_components.append(component_name)
        
        return {
            "early": early_components,
            "middle": middle_components,
            "late": late_components,
            "flow_pattern": self._characterize_flow_pattern(early_components, middle_components, late_components)
        }
    
    def _characterize_flow_pattern(
        self,
        early: List[str],
        middle: List[str],
        late: List[str]
    ) -> str:
        """Characterize the information flow pattern."""
        if len(early) > len(middle) and len(early) > len(late):
            return "early_binding"
        elif len(late) > len(middle) and len(late) > len(early):
            return "late_binding"
        elif len(middle) > len(early) and len(middle) > len(late):
            return "middle_processing"
        elif len(early) > 0 and len(late) > 0 and len(middle) == 0:
            return "skip_connection"
        else:
            return "distributed"
    
    def compare_causal_traces(
        self,
        traces: List[CausalTraceResult],
        component_names: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Compare causal traces across multiple examples.
        
        Args:
            traces: List of causal trace results
            component_names: Specific components to compare
            
        Returns:
            Dictionary with component names and their effects across traces
        """
        if not traces:
            return {}
        
        # Get all component names if not specified
        if component_names is None:
            all_components = set()
            for trace in traces:
                all_components.update(trace.trace_results.keys())
            component_names = sorted(list(all_components))
        
        comparison = {}
        for component in component_names:
            effects = []
            for trace in traces:
                effect = trace.trace_results.get(component, {}).get("restoration_effect", 0.0)
                effects.append(effect)
            comparison[component] = effects
        
        return comparison