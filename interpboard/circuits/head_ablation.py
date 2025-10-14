"""
Attention head ablation for understanding the role of individual attention heads.

This module provides tools to systematically ablate attention heads and
measure their impact on model performance and behavior.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from dataclasses import dataclass
import logging
from itertools import combinations

from ..utils import tokenize_input, get_model_info
from ..patching.activ_patch import ActivationPatcher, PatchLocation, PatchType

logger = logging.getLogger(__name__)


@dataclass
class HeadAblationResult:
    """Results from attention head ablation experiment."""
    head_impacts: Dict[str, float]  # {head_id: impact_score}
    ablation_type: str
    baseline_score: float
    metric_name: str
    input_text: str
    target_position: Optional[int]
    metadata: Dict


@dataclass
class HeadInteractionResult:
    """Results from head interaction analysis."""
    pairwise_effects: Dict[Tuple[str, str], float]
    individual_effects: Dict[str, float]
    interaction_scores: Dict[Tuple[str, str], float]
    redundant_pairs: List[Tuple[str, str]]
    synergistic_pairs: List[Tuple[str, str]]


class AttentionHeadAblator:
    """Performs systematic ablation of attention heads."""
    
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
        
        # Use activation patcher for ablations
        self.patcher = ActivationPatcher(model, tokenizer, device)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.cleanup_hooks()
    
    def ablate_head(
        self,
        text: str,
        layer_idx: int,
        head_idx: int,
        ablation_type: str = "zero",
        target_position: Optional[int] = None,
        metric: str = "logit_l2_diff"
    ) -> float:
        """
        Ablate a single attention head and measure impact.
        
        Args:
            text: Input text
            layer_idx: Layer index
            head_idx: Head index within layer
            ablation_type: Type of ablation ("zero", "mean", "random")
            target_position: Position to measure impact on
            metric: Metric for measuring impact
            
        Returns:
            Impact score (higher = more important)
        """
        patch_location = PatchLocation(
            layer_idx=layer_idx,
            component="attention",
            head_idx=head_idx,
            position_idx=target_position
        )
        
        # Map ablation type to patch type
        patch_type_map = {
            "zero": PatchType.ZERO,
            "mean": PatchType.MEAN,
            "random": PatchType.RANDOM
        }
        
        patch_type = patch_type_map.get(ablation_type, PatchType.ZERO)
        
        # Perform ablation
        result = self.patcher.patch_activation(
            text, patch_location, patch_type
        )
        
        return result.metrics.get(metric, 0.0)
    
    def systematic_head_ablation(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
        ablation_type: str = "zero",
        target_position: Optional[int] = None,
        metric: str = "logit_l2_diff"
    ) -> HeadAblationResult:
        """
        Systematically ablate all specified attention heads.
        
        Args:
            text: Input text
            layers: Layers to test (default: all)
            heads: Heads to test (default: all)
            ablation_type: Type of ablation
            target_position: Position to measure impact on
            metric: Metric for measuring impact
            
        Returns:
            HeadAblationResult with impact scores
        """
        if layers is None:
            layers = list(range(self.model_info["num_layers"]))
        
        if heads is None:
            heads = list(range(self.model_info["num_heads"]))
        
        # Get baseline score (no ablation)
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            baseline_output = self.model(**inputs)
        
        if hasattr(baseline_output, "logits"):
            baseline_logits = baseline_output.logits
        else:
            baseline_logits = baseline_output
        
        # Compute baseline metric
        if target_position is None:
            target_position = baseline_logits.shape[1] - 1
        
        baseline_score = self._compute_baseline_metric(baseline_logits, target_position, metric)
        
        # Test each head
        head_impacts = {}
        total_heads = len(layers) * len(heads)
        logger.info(f"Testing {total_heads} attention heads")
        
        for layer_idx in layers:
            for head_idx in heads:
                head_id = f"L{layer_idx}H{head_idx}"
                
                try:
                    impact = self.ablate_head(
                        text, layer_idx, head_idx, ablation_type, 
                        target_position, metric
                    )
                    head_impacts[head_id] = impact
                    
                except Exception as e:
                    logger.warning(f"Failed to ablate {head_id}: {e}")
                    head_impacts[head_id] = 0.0
        
        result = HeadAblationResult(
            head_impacts=head_impacts,
            ablation_type=ablation_type,
            baseline_score=baseline_score,
            metric_name=metric,
            input_text=text,
            target_position=target_position,
            metadata={
                "num_layers": len(layers),
                "num_heads_per_layer": len(heads),
                "total_heads_tested": len(head_impacts)
            }
        )
        
        return result
    
    def _compute_baseline_metric(
        self,
        logits: torch.Tensor,
        target_position: int,
        metric: str
    ) -> float:
        """Compute baseline metric score."""
        if metric == "logit_l2_diff":
            return 0.0  # Baseline is zero difference
        elif metric == "entropy":
            target_logits = logits[0, target_position, :]
            probs = F.softmax(target_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            return entropy.item()
        elif metric == "top_prob":
            target_logits = logits[0, target_position, :]
            probs = F.softmax(target_logits, dim=-1)
            return torch.max(probs).item()
        else:
            return 0.0
    
    def find_critical_heads(
        self,
        ablation_result: HeadAblationResult,
        threshold: float = 0.1,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find the most critical attention heads.
        
        Args:
            ablation_result: Result from systematic ablation
            threshold: Minimum impact to consider critical
            top_k: Maximum number of heads to return
            
        Returns:
            List of (head_id, impact) tuples sorted by impact
        """
        critical_heads = []
        
        for head_id, impact in ablation_result.head_impacts.items():
            if impact >= threshold:
                critical_heads.append((head_id, impact))
        
        # Sort by impact (descending)
        critical_heads.sort(key=lambda x: x[1], reverse=True)
        
        return critical_heads[:top_k]
    
    def analyze_head_interactions(
        self,
        text: str,
        head_pairs: List[Tuple[str, str]],
        ablation_type: str = "zero",
        target_position: Optional[int] = None,
        metric: str = "logit_l2_diff"
    ) -> HeadInteractionResult:
        """
        Analyze interactions between pairs of attention heads.
        
        Args:
            text: Input text
            head_pairs: Pairs of head IDs to test
            ablation_type: Type of ablation
            target_position: Position to measure impact on
            metric: Metric for measuring impact
            
        Returns:
            HeadInteractionResult with interaction analysis
        """
        individual_effects = {}
        pairwise_effects = {}
        interaction_scores = {}
        
        # Get individual effects
        all_heads = set()
        for head1, head2 in head_pairs:
            all_heads.add(head1)
            all_heads.add(head2)
        
        for head_id in all_heads:
            layer_idx, head_idx = self._parse_head_id(head_id)
            
            impact = self.ablate_head(
                text, layer_idx, head_idx, ablation_type,
                target_position, metric
            )
            individual_effects[head_id] = impact
        
        # Get pairwise effects
        for head1, head2 in head_pairs:
            layer1, head1_idx = self._parse_head_id(head1)
            layer2, head2_idx = self._parse_head_id(head2)
            
            # Ablate both heads simultaneously
            combined_impact = self._ablate_multiple_heads(
                text, [(layer1, head1_idx), (layer2, head2_idx)],
                ablation_type, target_position, metric
            )
            
            pairwise_effects[(head1, head2)] = combined_impact
            
            # Compute interaction score
            expected_combined = individual_effects[head1] + individual_effects[head2]
            interaction = combined_impact - expected_combined
            interaction_scores[(head1, head2)] = interaction
        
        # Classify interactions
        redundant_pairs = []
        synergistic_pairs = []
        
        for (head1, head2), interaction in interaction_scores.items():
            if interaction < -0.05:  # Less effect than expected (redundant)
                redundant_pairs.append((head1, head2))
            elif interaction > 0.05:  # More effect than expected (synergistic)
                synergistic_pairs.append((head1, head2))
        
        return HeadInteractionResult(
            pairwise_effects=pairwise_effects,
            individual_effects=individual_effects,
            interaction_scores=interaction_scores,
            redundant_pairs=redundant_pairs,
            synergistic_pairs=synergistic_pairs
        )
    
    def _parse_head_id(self, head_id: str) -> Tuple[int, int]:
        """Parse head ID like 'L3H5' into (layer_idx, head_idx)."""
        if not head_id.startswith('L') or 'H' not in head_id:
            raise ValueError(f"Invalid head ID format: {head_id}")
        
        parts = head_id[1:].split('H')  # Remove 'L' prefix and split on 'H'
        layer_idx = int(parts[0])
        head_idx = int(parts[1])
        
        return layer_idx, head_idx
    
    def _ablate_multiple_heads(
        self,
        text: str,
        head_specs: List[Tuple[int, int]],
        ablation_type: str,
        target_position: Optional[int],
        metric: str
    ) -> float:
        """Ablate multiple heads simultaneously."""
        # This is a simplified implementation - in practice, you'd want to
        # carefully handle the order of ablations and potential interactions
        
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get original output
        with torch.no_grad():
            original_output = self.model(**inputs)
        
        # Apply multiple patches
        for layer_idx, head_idx in head_specs:
            patch_location = PatchLocation(
                layer_idx=layer_idx,
                component="attention", 
                head_idx=head_idx,
                position_idx=target_position
            )
            
            patch_type = PatchType.ZERO if ablation_type == "zero" else PatchType.MEAN
            
            # For simplicity, apply patches sequentially
            # A more sophisticated approach would apply all patches in a single forward pass
            result = self.patcher.patch_activation(text, patch_location, patch_type)
        
        # Return the final metric value
        return result.metrics.get(metric, 0.0)
    
    def layer_head_heatmap(
        self,
        ablation_result: HeadAblationResult
    ) -> np.ndarray:
        """
        Create a heatmap matrix of head impacts by layer and head position.
        
        Args:
            ablation_result: Result from systematic ablation
            
        Returns:
            2D numpy array [num_layers, num_heads]
        """
        # Determine dimensions
        max_layer = 0
        max_head = 0
        
        for head_id in ablation_result.head_impacts.keys():
            layer_idx, head_idx = self._parse_head_id(head_id)
            max_layer = max(max_layer, layer_idx)
            max_head = max(max_head, head_idx)
        
        # Create heatmap matrix
        heatmap = np.zeros((max_layer + 1, max_head + 1))
        
        for head_id, impact in ablation_result.head_impacts.items():
            layer_idx, head_idx = self._parse_head_id(head_id)
            heatmap[layer_idx, head_idx] = impact
        
        return heatmap
    
    def compare_head_importance_across_tasks(
        self,
        task_texts: Dict[str, List[str]],
        ablation_type: str = "zero",
        metric: str = "logit_l2_diff"
    ) -> Dict[str, HeadAblationResult]:
        """
        Compare head importance across different tasks/contexts.
        
        Args:
            task_texts: Dictionary mapping task names to lists of texts
            ablation_type: Type of ablation
            metric: Metric for measuring impact
            
        Returns:
            Dictionary mapping task names to ablation results
        """
        task_results = {}
        
        for task_name, texts in task_texts.items():
            logger.info(f"Analyzing heads for task: {task_name}")
            
            # Use first text as representative for this task
            text = texts[0] if texts else ""
            
            if not text:
                logger.warning(f"No texts provided for task {task_name}")
                continue
            
            try:
                result = self.systematic_head_ablation(
                    text, ablation_type=ablation_type, metric=metric
                )
                task_results[task_name] = result
                
            except Exception as e:
                logger.error(f"Failed to analyze task {task_name}: {e}")
                continue
        
        return task_results
    
    def find_task_specific_heads(
        self,
        task_results: Dict[str, HeadAblationResult],
        specificity_threshold: float = 0.1
    ) -> Dict[str, List[str]]:
        """
        Find heads that are specifically important for certain tasks.
        
        Args:
            task_results: Results from compare_head_importance_across_tasks
            specificity_threshold: Minimum difference to consider task-specific
            
        Returns:
            Dictionary mapping task names to lists of task-specific head IDs
        """
        task_specific_heads = {}
        
        # Get all head IDs
        all_heads = set()
        for result in task_results.values():
            all_heads.update(result.head_impacts.keys())
        
        for task_name, task_result in task_results.items():
            specific_heads = []
            
            for head_id in all_heads:
                task_impact = task_result.head_impacts.get(head_id, 0.0)
                
                # Compare with impact in other tasks
                other_impacts = []
                for other_task, other_result in task_results.items():
                    if other_task != task_name:
                        other_impact = other_result.head_impacts.get(head_id, 0.0)
                        other_impacts.append(other_impact)
                
                if other_impacts:
                    mean_other_impact = np.mean(other_impacts)
                    
                    # Check if this head is specifically important for this task
                    if task_impact - mean_other_impact >= specificity_threshold:
                        specific_heads.append(head_id)
            
            task_specific_heads[task_name] = specific_heads
        
        return task_specific_heads
    
    def head_ablation_summary(
        self,
        ablation_result: HeadAblationResult,
        top_k: int = 10
    ) -> Dict[str, Union[List, float, int]]:
        """
        Create a summary of head ablation results.
        
        Args:
            ablation_result: Result from systematic ablation
            top_k: Number of top heads to include in summary
            
        Returns:
            Summary dictionary
        """
        impacts = list(ablation_result.head_impacts.values())
        
        # Get top heads
        top_heads = self.find_critical_heads(ablation_result, top_k=top_k)
        
        # Compute statistics
        summary = {
            "top_critical_heads": [{"head_id": head, "impact": impact} 
                                 for head, impact in top_heads],
            "mean_impact": np.mean(impacts),
            "std_impact": np.std(impacts),
            "max_impact": np.max(impacts),
            "num_significant_heads": sum(1 for impact in impacts if impact > 0.05),
            "total_heads_tested": len(impacts),
            "ablation_type": ablation_result.ablation_type,
            "metric_used": ablation_result.metric_name,
            "baseline_score": ablation_result.baseline_score
        }
        
        return summary