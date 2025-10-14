"""
Gradient-based attribution methods for understanding token importance.

This module implements various gradient-based attribution methods including
vanilla gradients, integrated gradients, and gradient x input.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from dataclasses import dataclass
import logging
from ..utils import tokenize_input, normalize_tensor, safe_divide
logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Results from an attribution method."""
    attributions: torch.Tensor  # [batch_size, seq_len] or [batch_size, seq_len, vocab_size]
    tokens: List[str]
    token_ids: torch.Tensor
    method_name: str
    target_token_id: Optional[int] = None
    target_position: Optional[int] = None
    metadata: Dict = None


class GradientAttributor:
    """Implements gradient-based attribution methods."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        # Ensure model is in eval mode but gradients are enabled
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(True)
    
    def vanilla_gradient(
        self,
        text: str,
        target_token_id: Optional[int] = None,
        target_position: Optional[int] = None,
        normalize: bool = True
    ) -> AttributionResult:
        """
        Compute vanilla gradients w.r.t. input embeddings.
        
        Args:
            text: Input text to analyze
            target_token_id: Target token ID to compute gradients for
            target_position: Position to compute gradients for (default: last)
            normalize: Whether to normalize gradients
            
        Returns:
            AttributionResult with gradient attributions
        """
        inputs = tokenize_input(text, self.tokenizer)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        
        # Get embeddings and enable gradients
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Ensure gradient is retained for non-leaf tensor
        
        # Forward pass
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        # Determine target
        seq_len = logits.shape[1]
        if target_position is None:
            target_position = seq_len - 1
        
        if target_token_id is None:
            # Use the predicted token
            target_token_id = torch.argmax(logits[0, target_position]).item()
        
        # Compute loss/score for target
        target_score = logits[0, target_position, target_token_id]
        
        # Backward pass
        target_score.backward(retain_graph=True)
        
        # Get gradients w.r.t. embeddings
        gradients = embeddings.grad  # [batch_size, seq_len, embedding_dim]
        
        if gradients is None:
            raise RuntimeError("Gradients are None. Make sure model parameters require gradients.")
        
        # Compute attribution scores (gradient norm for each token)
        attributions = torch.norm(gradients, dim=-1)  # [batch_size, seq_len]
        
        if normalize:
            attributions = normalize_tensor(attributions, dim=-1)
        
        # Convert to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return AttributionResult(
            attributions=attributions[0],  # Remove batch dimension
            tokens=tokens,
            token_ids=input_ids[0],
            method_name="vanilla_gradient",
            target_token_id=target_token_id,
            target_position=target_position,
            metadata={
                "target_score": target_score.item(),
                "normalized": normalize
            }
        )
    
    def gradient_x_input(
        self,
        text: str,
        target_token_id: Optional[int] = None,
        target_position: Optional[int] = None,
        normalize: bool = True
    ) -> AttributionResult:
        """
        Compute gradient × input attribution.
        
        Args:
            text: Input text to analyze
            target_token_id: Target token ID
            target_position: Position to compute gradients for
            normalize: Whether to normalize attributions
            
        Returns:
            AttributionResult with gradient × input attributions
        """
        inputs = tokenize_input(text, self.tokenizer)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        
        # Get embeddings
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Ensure gradient is retained
        
        # Forward pass
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
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
        target_score.backward(retain_graph=True)
        
        # Get gradients
        gradients = embeddings.grad  # [batch_size, seq_len, embedding_dim]
        
        if gradients is None:
            raise RuntimeError("Gradients are None. Make sure embeddings retain gradients.")
        
        # Compute gradient × input
        grad_x_input = gradients * embeddings  # Element-wise multiplication
        
        # Sum over embedding dimension
        attributions = torch.sum(grad_x_input, dim=-1)  # [batch_size, seq_len]
        
        if normalize:
            attributions = normalize_tensor(attributions, dim=-1)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return AttributionResult(
            attributions=attributions[0],
            tokens=tokens,
            token_ids=input_ids[0],
            method_name="gradient_x_input",
            target_token_id=target_token_id,
            target_position=target_position,
            metadata={
                "target_score": target_score.item(),
                "normalized": normalize
            }
        )
    
    def integrated_gradients(
        self,
        text: str,
        target_token_id: Optional[int] = None,
        target_position: Optional[int] = None,
        baseline: str = "zero",
        steps: int = 50,
        normalize: bool = True
    ) -> AttributionResult:
        """
        Compute integrated gradients attribution.
        
        Args:
            text: Input text to analyze
            target_token_id: Target token ID
            target_position: Position to compute gradients for
            baseline: Baseline type ("zero", "mask", "random")
            steps: Number of integration steps
            normalize: Whether to normalize attributions
            
        Returns:
            AttributionResult with integrated gradients
        """
        inputs = tokenize_input(text, self.tokenizer)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        
        # Get original embeddings
        original_embeddings = self.model.get_input_embeddings()(input_ids)
        
        # Create baseline embeddings
        if baseline == "zero":
            baseline_embeddings = torch.zeros_like(original_embeddings)
        elif baseline == "mask":
            mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
            mask_ids = torch.full_like(input_ids, mask_token_id)
            baseline_embeddings = self.model.get_input_embeddings()(mask_ids)
        elif baseline == "random":
            baseline_embeddings = torch.randn_like(original_embeddings)
        else:
            raise ValueError(f"Unknown baseline type: {baseline}")
        
        # Determine target
        with torch.no_grad():
            outputs = self.model(inputs_embeds=original_embeddings, attention_mask=attention_mask)
            logits = outputs.logits
            seq_len = logits.shape[1]
            
            if target_position is None:
                target_position = seq_len - 1
            
            if target_token_id is None:
                target_token_id = torch.argmax(logits[0, target_position]).item()
        
        # Integrate gradients
        integrated_gradients = torch.zeros_like(original_embeddings)
        
        for step in range(steps):
            # Interpolate between baseline and original
            alpha = step / (steps - 1) if steps > 1 else 1.0
            interpolated_embeddings = baseline_embeddings + alpha * (original_embeddings - baseline_embeddings)
            interpolated_embeddings.requires_grad_(True)
            interpolated_embeddings.retain_grad()  # Ensure gradients are retained
            
            # Forward pass
            outputs = self.model(
                inputs_embeds=interpolated_embeddings,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            # Compute target score
            target_score = logits[0, target_position, target_token_id]
            
            # Backward pass
            if interpolated_embeddings.grad is not None:
                interpolated_embeddings.grad.zero_()
            
            target_score.backward(retain_graph=True)
            
            # Check if gradients exist
            if interpolated_embeddings.grad is not None:
                integrated_gradients += interpolated_embeddings.grad / steps
            else:
                logger.warning(f"No gradients found for integration step {step}")
                # Use zero gradients for this step
                integrated_gradients += torch.zeros_like(interpolated_embeddings) / steps
        
        # Compute final attributions
        diff = original_embeddings - baseline_embeddings
        attributions = torch.sum(integrated_gradients * diff, dim=-1)  # [batch_size, seq_len]
        
        if normalize:
            attributions = normalize_tensor(attributions, dim=-1)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return AttributionResult(
            attributions=attributions[0],
            tokens=tokens,
            token_ids=input_ids[0],
            method_name="integrated_gradients",
            target_token_id=target_token_id,
            target_position=target_position,
            metadata={
                "baseline": baseline,
                "steps": steps,
                "normalized": normalize
            }
        )
    
    def layerwise_relevance_propagation(
        self,
        text: str,
        target_token_id: Optional[int] = None,
        target_position: Optional[int] = None,
        epsilon: float = 1e-8
    ) -> AttributionResult:
        """
        Compute Layer-wise Relevance Propagation (LRP) attribution.
        Note: This is a simplified implementation.
        
        Args:
            text: Input text to analyze
            target_token_id: Target token ID
            target_position: Position to compute relevance for
            epsilon: Small value for numerical stability
            
        Returns:
            AttributionResult with LRP attributions
        """
        # For simplicity, this implements a basic form of LRP
        # A full implementation would require modifying the forward pass
        
        inputs = tokenize_input(text, self.tokenizer)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        
        # Get embeddings
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Ensure gradients are retained
        
        # Forward pass
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Determine target
        seq_len = logits.shape[1]
        if target_position is None:
            target_position = seq_len - 1
        
        if target_token_id is None:
            target_token_id = torch.argmax(logits[0, target_position]).item()
        
        # Compute target score
        target_score = logits[0, target_position, target_token_id]
        
        # Simple LRP: use gradients with epsilon rule
        target_score.backward(retain_graph=True)
        gradients = embeddings.grad
        
        if gradients is None:
            raise RuntimeError("Gradients are None. Cannot compute LRP.")
        
        # LRP epsilon rule: R = (input * gradient) / (input + epsilon * sign(input))
        denominator = embeddings + epsilon * torch.sign(embeddings)
        relevance = safe_divide(embeddings * gradients, denominator)
        
        # Sum over embedding dimension
        attributions = torch.sum(relevance, dim=-1)  # [batch_size, seq_len]
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return AttributionResult(
            attributions=attributions[0],
            tokens=tokens,
            token_ids=input_ids[0],
            method_name="lrp",
            target_token_id=target_token_id,
            target_position=target_position,
            metadata={
                "epsilon": epsilon,
                "target_score": target_score.item()
            }
        )
    
    def compute_all_attributions(
        self,
        text: str,
        target_token_id: Optional[int] = None,
        target_position: Optional[int] = None,
        methods: Optional[List[str]] = None
    ) -> Dict[str, AttributionResult]:
        """
        Compute multiple attribution methods for comparison.
        
        Args:
            text: Input text to analyze
            target_token_id: Target token ID
            target_position: Position to compute attributions for
            methods: List of methods to compute
            
        Returns:
            Dictionary mapping method names to AttributionResult
        """
        if methods is None:
            methods = ["vanilla_gradient", "gradient_x_input", "integrated_gradients"]
        
        results = {}
        
        for method in methods:
            try:
                if method == "vanilla_gradient":
                    result = self.vanilla_gradient(text, target_token_id, target_position)
                elif method == "gradient_x_input":
                    result = self.gradient_x_input(text, target_token_id, target_position)
                elif method == "integrated_gradients":
                    result = self.integrated_gradients(text, target_token_id, target_position)
                elif method == "lrp":
                    result = self.layerwise_relevance_propagation(text, target_token_id, target_position)
                else:
                    logger.warning(f"Unknown attribution method: {method}")
                    continue
                
                results[method] = result
                
            except Exception as e:
                logger.error(f"Error computing {method}: {e}")
                continue
        
        return results
    
    def compare_attributions(
        self,
        attribution_results: Dict[str, AttributionResult],
        correlation_method: str = "pearson"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different attribution methods.
        
        Args:
            attribution_results: Dictionary of attribution results
            correlation_method: Method for computing correlation
            
        Returns:
            Correlation matrix between methods
        """
        methods = list(attribution_results.keys())
        correlations = {}
        
        for method1 in methods:
            correlations[method1] = {}
            attr1 = attribution_results[method1].attributions
            
            for method2 in methods:
                attr2 = attribution_results[method2].attributions
                
                if correlation_method == "pearson":
                    correlation = torch.corrcoef(torch.stack([attr1, attr2]))[0, 1]
                elif correlation_method == "spearman":
                    # Simple rank correlation
                    rank1 = torch.argsort(torch.argsort(attr1))
                    rank2 = torch.argsort(torch.argsort(attr2))
                    correlation = torch.corrcoef(torch.stack([rank1.float(), rank2.float()]))[0, 1]
                else:
                    raise ValueError(f"Unknown correlation method: {correlation_method}")
                
                correlations[method1][method2] = correlation.item() if not torch.isnan(correlation) else 0.0
        
        return correlations