"""
Attention-based attribution methods for understanding token importance.

This module implements attention rollout, attention flow, and other
attention-based attribution techniques.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from dataclasses import dataclass
import logging

from ..utils import tokenize_input, compute_attention_rollout, normalize_tensor
from .gradients import AttributionResult

logger = logging.getLogger(__name__)


class AttentionAttributor:
    """Implements attention-based attribution methods."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    def attention_rollout(
        self,
        text: str,
        discard_ratio: float = 0.9,
        head_fusion: str = "mean",
        normalize: bool = True
    ) -> AttributionResult:
        """
        Compute attention rollout following Abnar & Zuidema (2020).
        
        Args:
            text: Input text to analyze
            discard_ratio: Ratio of attention to discard in each step
            head_fusion: How to combine attention heads ("mean", "max", "min")
            normalize: Whether to normalize final attributions
            
        Returns:
            AttributionResult with attention rollout scores
        """
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions  # List of [batch, heads, seq_len, seq_len]
        
        if attentions is None or len(attentions) == 0:
            raise RuntimeError("No attention weights returned by model. Make sure model supports output_attentions=True")
        
        if attentions[0] is None:
            raise RuntimeError("Attention weights are None. Model may not support attention output.")
        
        seq_len = attentions[0].shape[-1]
        
        # Stack attention weights: [batch, layers, heads, seq_len, seq_len]
        attention_weights = torch.stack(attentions, dim=1)
        
        # Fuse attention heads
        if head_fusion == "mean":
            fused_attention = attention_weights.mean(dim=2)
        elif head_fusion == "max":
            fused_attention = attention_weights.max(dim=2)[0]
        elif head_fusion == "min":
            fused_attention = attention_weights.min(dim=2)[0]
        else:
            raise ValueError(f"Unknown head fusion method: {head_fusion}")
        
        # Add residual connections
        residual_att = torch.eye(seq_len, device=self.device)
        residual_att = residual_att.expand_as(fused_attention)
        aug_att_mat = fused_attention + residual_att
        
        # Normalize
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)
        
        # Rollout
        joint_attentions = aug_att_mat[0, 0]  # Start with first layer
        
        for i in range(1, aug_att_mat.shape[1]):
            joint_attentions = torch.matmul(aug_att_mat[0, i], joint_attentions)
        
        # Get attributions for each token (how much it attends to others)
        attributions = joint_attentions.sum(dim=0)  # [seq_len]
        
        if normalize:
            attributions = normalize_tensor(attributions, dim=0)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return AttributionResult(
            attributions=attributions,
            tokens=tokens,
            token_ids=inputs["input_ids"][0],
            method_name="attention_rollout",
            metadata={
                "discard_ratio": discard_ratio,
                "head_fusion": head_fusion,
                "normalized": normalize,
                "attention_matrix": joint_attentions.cpu()
            }
        )
    
    def attention_flow(
        self,
        text: str,
        target_position: Optional[int] = None,
        head_fusion: str = "mean",
        layer_fusion: str = "mean"
    ) -> AttributionResult:
        """
        Compute attention flow to a target position.
        
        Args:
            text: Input text to analyze
            target_position: Position to compute flow to (default: last)
            head_fusion: How to combine attention heads
            layer_fusion: How to combine layers ("mean", "last", "weighted")
            
        Returns:
            AttributionResult with attention flow scores
        """
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions
        
        if attentions is None or len(attentions) == 0:
            raise RuntimeError("No attention weights returned by model")
        
        seq_len = attentions[0].shape[-1]
        
        if target_position is None:
            target_position = seq_len - 1
        
        # Extract attention to target position
        target_attentions = []
        
        for layer_attention in attentions:
            # layer_attention: [batch, heads, seq_len, seq_len]
            target_att = layer_attention[0, :, target_position, :]  # [heads, seq_len]
            
            # Fuse heads
            if head_fusion == "mean":
                fused = target_att.mean(dim=0)
            elif head_fusion == "max":
                fused = target_att.max(dim=0)[0]
            elif head_fusion == "sum":
                fused = target_att.sum(dim=0)
            else:
                raise ValueError(f"Unknown head fusion: {head_fusion}")
            
            target_attentions.append(fused)
        
        # Stack layers: [layers, seq_len]
        layer_attentions = torch.stack(target_attentions, dim=0)
        
        # Fuse layers
        if layer_fusion == "mean":
            attributions = layer_attentions.mean(dim=0)
        elif layer_fusion == "last":
            attributions = layer_attentions[-1]
        elif layer_fusion == "weighted":
            # Weight later layers more heavily
            weights = torch.linspace(0.1, 1.0, len(attentions), device=self.device)
            weights = weights / weights.sum()
            attributions = (layer_attentions * weights.unsqueeze(1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown layer fusion: {layer_fusion}")
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return AttributionResult(
            attributions=attributions,
            tokens=tokens,
            token_ids=inputs["input_ids"][0],
            method_name="attention_flow",
            target_position=target_position,
            metadata={
                "head_fusion": head_fusion,
                "layer_fusion": layer_fusion,
                "layer_attentions": layer_attentions.cpu()
            }
        )
    
    def head_specific_attribution(
        self,
        text: str,
        layer_idx: int,
        head_idx: int,
        target_position: Optional[int] = None
    ) -> AttributionResult:
        """
        Get attribution from a specific attention head.
        
        Args:
            text: Input text to analyze
            layer_idx: Layer index
            head_idx: Head index within the layer
            target_position: Position to compute attribution for
            
        Returns:
            AttributionResult for the specific head
        """
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions
        
        if layer_idx >= len(attentions):
            raise ValueError(f"Layer index {layer_idx} >= number of layers {len(attentions)}")
        
        layer_attention = attentions[layer_idx]  # [batch, heads, seq_len, seq_len]
        
        if head_idx >= layer_attention.shape[1]:
            raise ValueError(f"Head index {head_idx} >= number of heads {layer_attention.shape[1]}")
        
        head_attention = layer_attention[0, head_idx]  # [seq_len, seq_len]
        
        if target_position is None:
            # Use the last position
            target_position = head_attention.shape[0] - 1
        
        # Get attention from all positions to target position
        attributions = head_attention[target_position, :]  # [seq_len]
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return AttributionResult(
            attributions=attributions,
            tokens=tokens,
            token_ids=inputs["input_ids"][0],
            method_name=f"head_L{layer_idx}H{head_idx}",
            target_position=target_position,
            metadata={
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "attention_matrix": head_attention.cpu()
            }
        )
    
    def multi_head_attribution(
        self,
        text: str,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        target_position: Optional[int] = None,
        aggregation: str = "mean"
    ) -> Dict[str, AttributionResult]:
        """
        Get attributions from multiple attention heads.
        
        Args:
            text: Input text to analyze
            layer_indices: Layers to analyze (default: all)
            head_indices: Heads to analyze (default: all)
            target_position: Position to compute attribution for
            aggregation: How to aggregate multiple heads
            
        Returns:
            Dictionary of AttributionResult for each head
        """
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions
        
        if layer_indices is None:
            layer_indices = list(range(len(attentions)))
        
        if head_indices is None:
            head_indices = list(range(attentions[0].shape[1]))
        
        results = {}
        all_attributions = []
        
        for layer_idx in layer_indices:
            for head_idx in head_indices:
                try:
                    result = self.head_specific_attribution(
                        text, layer_idx, head_idx, target_position
                    )
                    head_name = f"L{layer_idx}H{head_idx}"
                    results[head_name] = result
                    all_attributions.append(result.attributions)
                    
                except Exception as e:
                    logger.warning(f"Failed to compute attribution for L{layer_idx}H{head_idx}: {e}")
                    continue
        
        # Add aggregated result
        if all_attributions:
            stacked_attributions = torch.stack(all_attributions, dim=0)
            
            if aggregation == "mean":
                agg_attributions = stacked_attributions.mean(dim=0)
            elif aggregation == "max":
                agg_attributions = stacked_attributions.max(dim=0)[0]
            elif aggregation == "sum":
                agg_attributions = stacked_attributions.sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            results["aggregated"] = AttributionResult(
                attributions=agg_attributions,
                tokens=tokens,
                token_ids=inputs["input_ids"][0],
                method_name=f"multi_head_{aggregation}",
                target_position=target_position,
                metadata={
                    "aggregation": aggregation,
                    "num_heads": len(all_attributions),
                    "layer_indices": layer_indices,
                    "head_indices": head_indices
                }
            )
        
        return results
    
    def attention_entropy(
        self,
        text: str,
        normalize: bool = True
    ) -> AttributionResult:
        """
        Compute attention entropy for each token position.
        Higher entropy means more diffuse attention.
        
        Args:
            text: Input text to analyze
            normalize: Whether to normalize entropy values
            
        Returns:
            AttributionResult with entropy scores
        """
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions
        seq_len = attentions[0].shape[-1]
        
        entropies = []
        
        for layer_attention in attentions:
            # layer_attention: [batch, heads, seq_len, seq_len]
            layer_entropies = []
            
            for head in range(layer_attention.shape[1]):
                head_attention = layer_attention[0, head]  # [seq_len, seq_len]
                
                # Compute entropy for each position's attention distribution
                head_entropy = -torch.sum(
                    head_attention * torch.log(head_attention + 1e-8), dim=-1
                )  # [seq_len]
                
                layer_entropies.append(head_entropy)
            
            # Average across heads
            layer_entropy = torch.stack(layer_entropies, dim=0).mean(dim=0)
            entropies.append(layer_entropy)
        
        # Average across layers
        attributions = torch.stack(entropies, dim=0).mean(dim=0)  # [seq_len]
        
        if normalize:
            attributions = normalize_tensor(attributions, dim=0)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return AttributionResult(
            attributions=attributions,
            tokens=tokens,
            token_ids=inputs["input_ids"][0],
            method_name="attention_entropy",
            metadata={
                "normalized": normalize,
                "layer_entropies": torch.stack(entropies, dim=0).cpu()
            }
        )
    
    def attention_based_gradient(
        self,
        text: str,
        target_token_id: Optional[int] = None,
        target_position: Optional[int] = None,
        layer_idx: Optional[int] = None
    ) -> AttributionResult:
        """
        Compute gradients with respect to attention weights.
        
        Args:
            text: Input text to analyze
            target_token_id: Target token ID
            target_position: Position to compute gradients for
            layer_idx: Specific layer to analyze (default: last)
            
        Returns:
            AttributionResult with attention-based gradients
        """
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Store attention weights with gradients
        attention_weights = []
        
        def attention_hook(module, input, output):
            # output is typically (context_layer, attention_probs, ...)
            if isinstance(output, tuple) and len(output) > 1:
                attn_probs = output[1]  # attention probabilities
                if attn_probs.requires_grad:
                    attention_weights.append(attn_probs)
            return output
        
        # Register hooks to capture attention weights
        hooks = []
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "dropout"):
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        try:
            # Forward pass
            outputs = self.model(**inputs, output_attentions=True)
            logits = outputs.logits
            
            # Determine target
            seq_len = logits.shape[1]
            if target_position is None:
                target_position = seq_len - 1
            
            if target_token_id is None:
                target_token_id = torch.argmax(logits[0, target_position]).item()
            
            # Compute target score
            target_score = logits[0, target_position, target_token_id]
            
            # Backward pass
            target_score.backward()
            
            # Get gradients from attention weights
            if layer_idx is None:
                layer_idx = len(attention_weights) - 1
            
            if layer_idx < len(attention_weights):
                attn_grad = attention_weights[layer_idx].grad
                if attn_grad is not None:
                    # Average over heads and get attribution for target position
                    attributions = attn_grad[0, :, target_position, :].mean(dim=0)
                else:
                    attributions = torch.zeros(seq_len, device=self.device)
            else:
                attributions = torch.zeros(seq_len, device=self.device)
            
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return AttributionResult(
            attributions=attributions,
            tokens=tokens,
            token_ids=inputs["input_ids"][0],
            method_name="attention_gradient",
            target_token_id=target_token_id,
            target_position=target_position,
            metadata={
                "layer_idx": layer_idx,
                "target_score": target_score.item()
            }
        )