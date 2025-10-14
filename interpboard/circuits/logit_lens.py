"""
Logit lens implementation for understanding hidden state representations.

The logit lens projects hidden states from intermediate layers through the 
final layer norm and unembedding matrix to see what tokens they predict
at each layer.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from dataclasses import dataclass
import logging

from ..utils import tokenize_input, get_model_info

logger = logging.getLogger(__name__)


@dataclass 
class LogitLensResult:
    """Results from logit lens analysis."""
    layer_predictions: Dict[int, torch.Tensor]  # layer_idx -> logits [seq_len, vocab_size]
    layer_probabilities: Dict[int, torch.Tensor]  # layer_idx -> probs [seq_len, vocab_size]
    top_tokens: Dict[int, List[List[str]]]  # layer_idx -> [seq_len][top_k]
    input_text: str
    tokens: List[str]
    metadata: Dict


class LogitLens:
    """Implements logit lens analysis for transformer models."""
    
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
        
        # Get components for logit lens
        self._setup_components()
    
    def _setup_components(self):
        """Setup model components needed for logit lens."""
        model_type = self.model.config.model_type.lower()
        
        if "gpt2" in model_type:
            self.ln_f = self.model.transformer.ln_f
            self.lm_head = self.model.lm_head
        elif "llama" in model_type:
            self.ln_f = self.model.model.norm
            self.lm_head = self.model.lm_head
        elif "mistral" in model_type:
            self.ln_f = self.model.model.norm  
            self.lm_head = self.model.lm_head
        else:
            # Try to find components automatically
            self.ln_f = None
            self.lm_head = None
            
            for name, module in self.model.named_modules():
                if "norm" in name.lower() and hasattr(module, "weight"):
                    self.ln_f = module
                elif "lm_head" in name.lower() or "head" in name.lower():
                    self.lm_head = module
            
            if self.ln_f is None or self.lm_head is None:
                logger.warning(f"Could not auto-detect components for {model_type}")
    
    def compute_logits_at_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        apply_ln: bool = True
    ) -> torch.Tensor:
        """
        Compute logits by passing hidden states through final components.
        
        Args:
            hidden_states: Hidden states from a layer [batch, seq_len, hidden_size]
            layer_idx: Layer index (for logging)
            apply_ln: Whether to apply layer normalization
            
        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        if self.ln_f is None or self.lm_head is None:
            raise ValueError("Model components not properly initialized")
        
        # Apply final layer norm if requested
        if apply_ln:
            normalized_states = self.ln_f(hidden_states)
        else:
            normalized_states = hidden_states
        
        # Apply language modeling head
        logits = self.lm_head(normalized_states)
        
        return logits
    
    def analyze(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        top_k: int = 10,
        include_probabilities: bool = True
    ) -> LogitLensResult:
        """
        Perform logit lens analysis on input text.
        
        Args:
            text: Input text to analyze
            layers: Specific layers to analyze (default: all)
            top_k: Number of top predictions to store
            include_probabilities: Whether to compute probabilities
            
        Returns:
            LogitLensResult with analysis results
        """
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Store hidden states from all layers
        hidden_states_cache = {}
        
        def make_cache_hook(layer_idx: int):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                hidden_states_cache[layer_idx] = hidden_states.detach()
            return hook
        
        # Register hooks for all transformer layers
        hooks = []
        layer_count = 0
        
        for name, module in self.model.named_modules():
            # Check if this is a transformer layer
            if any(layer_type in name for layer_type in ["layer", "block", "h."]):
                # Only hook the main layer modules, not sub-components
                if "." in name:
                    continue
                    
                hook = module.register_forward_hook(make_cache_hook(layer_count))
                hooks.append(hook)
                layer_count += 1
        
        try:
            # Forward pass to collect hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Use output hidden states if hooks didn't work
            if not hidden_states_cache and hasattr(outputs, 'hidden_states'):
                for i, hidden_state in enumerate(outputs.hidden_states):
                    hidden_states_cache[i] = hidden_state
        
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        if not hidden_states_cache:
            raise RuntimeError("Failed to collect hidden states")
        
        # Determine which layers to analyze
        if layers is None:
            layers = sorted(hidden_states_cache.keys())
        
        # Analyze each layer
        layer_predictions = {}
        layer_probabilities = {}
        top_tokens = {}
        
        for layer_idx in layers:
            if layer_idx not in hidden_states_cache:
                logger.warning(f"Layer {layer_idx} not found in cache")
                continue
            
            hidden_states = hidden_states_cache[layer_idx]
            
            # Compute logits at this layer
            logits = self.compute_logits_at_layer(hidden_states, layer_idx)
            layer_predictions[layer_idx] = logits[0]  # Remove batch dimension
            
            # Compute probabilities if requested
            if include_probabilities:
                probs = F.softmax(logits, dim=-1)
                layer_probabilities[layer_idx] = probs[0]
            
            # Get top-k tokens for each position
            top_k_logits, top_k_indices = torch.topk(logits[0], top_k, dim=-1)
            
            position_tokens = []
            for pos in range(top_k_indices.shape[0]):
                pos_tokens = []
                for k in range(top_k):
                    token_id = top_k_indices[pos, k].item()
                    token = self.tokenizer.decode([token_id])
                    pos_tokens.append(token)
                position_tokens.append(pos_tokens)
            
            top_tokens[layer_idx] = position_tokens
        
        # Get input tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        result = LogitLensResult(
            layer_predictions=layer_predictions,
            layer_probabilities=layer_probabilities,
            top_tokens=top_tokens,
            input_text=text,
            tokens=tokens,
            metadata={
                "num_layers_analyzed": len(layers),
                "layers_analyzed": layers,
                "top_k": top_k,
                "sequence_length": len(tokens),
                "model_type": self.model.config.model_type
            }
        )
        
        return result
    
    def compare_predictions(
        self,
        result: LogitLensResult,
        position: int,
        layers: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, Union[str, float]]]:
        """
        Compare predictions across layers for a specific position.
        
        Args:
            result: LogitLensResult from analysis
            position: Token position to analyze
            layers: Specific layers to compare
            
        Returns:
            Dictionary mapping layer to prediction info
        """
        if layers is None:
            layers = sorted(result.layer_predictions.keys())
        
        comparisons = {}
        
        for layer_idx in layers:
            if layer_idx not in result.layer_predictions:
                continue
            
            logits = result.layer_predictions[layer_idx]
            
            if position >= logits.shape[0]:
                continue
            
            # Get top prediction
            top_logit, top_idx = torch.topk(logits[position], 1)
            top_token = self.tokenizer.decode([top_idx.item()])
            
            # Get probability if available
            prob = None
            if layer_idx in result.layer_probabilities:
                probs = result.layer_probabilities[layer_idx]
                prob = probs[position, top_idx].item()
            
            comparisons[layer_idx] = {
                "token": top_token,
                "logit": top_logit.item(),
                "probability": prob,
                "token_id": top_idx.item()
            }
        
        return comparisons
    
    def find_prediction_changes(
        self,
        result: LogitLensResult,
        position: int,
        threshold: float = 0.1
    ) -> List[Tuple[int, str, str]]:
        """
        Find layers where predictions change significantly.
        
        Args:
            result: LogitLensResult from analysis
            position: Token position to analyze
            threshold: Minimum probability change to consider significant
            
        Returns:
            List of (layer_idx, old_token, new_token) tuples
        """
        layers = sorted(result.layer_predictions.keys())
        changes = []
        
        if not result.layer_probabilities:
            logger.warning("Probabilities not available for change detection")
            return changes
        
        prev_top_token = None
        prev_prob = 0.0
        
        for layer_idx in layers:
            if layer_idx not in result.layer_probabilities:
                continue
            
            probs = result.layer_probabilities[layer_idx]
            
            if position >= probs.shape[0]:
                continue
            
            # Get current top prediction
            top_prob, top_idx = torch.topk(probs[position], 1)
            top_token = self.tokenizer.decode([top_idx.item()])
            curr_prob = top_prob.item()
            
            # Check for significant change
            if prev_top_token is not None:
                if (top_token != prev_top_token and 
                    abs(curr_prob - prev_prob) >= threshold):
                    changes.append((layer_idx, prev_top_token, top_token))
            
            prev_top_token = top_token
            prev_prob = curr_prob
        
        return changes
    
    def get_prediction_confidence(
        self,
        result: LogitLensResult,
        layers: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Get prediction confidence (entropy) for each layer.
        
        Args:
            result: LogitLensResult from analysis
            layers: Specific layers to analyze
            
        Returns:
            Dictionary mapping layer to confidence scores [seq_len]
        """
        if layers is None:
            layers = sorted(result.layer_probabilities.keys())
        
        confidences = {}
        
        for layer_idx in layers:
            if layer_idx not in result.layer_probabilities:
                continue
            
            probs = result.layer_probabilities[layer_idx]
            
            # Compute entropy (negative entropy = confidence)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            confidence = -entropy  # Higher is more confident
            
            confidences[layer_idx] = confidence
        
        return confidences
    
    def analyze_convergence(
        self,
        result: LogitLensResult,
        position: int,
        final_prediction: Optional[str] = None
    ) -> Dict[str, Union[int, List[float]]]:
        """
        Analyze how predictions converge to final prediction.
        
        Args:
            result: LogitLensResult from analysis
            position: Token position to analyze
            final_prediction: Expected final prediction (if known)
            
        Returns:
            Convergence analysis results
        """
        layers = sorted(result.layer_predictions.keys())
        
        if final_prediction is None:
            # Use prediction from last layer
            last_layer = max(layers)
            if last_layer in result.layer_predictions:
                logits = result.layer_predictions[last_layer]
                if position < logits.shape[0]:
                    top_idx = torch.argmax(logits[position])
                    final_prediction = self.tokenizer.decode([top_idx.item()])
        
        if final_prediction is None:
            logger.warning("Could not determine final prediction")
            return {}
        
        # Get token ID for final prediction
        final_token_ids = self.tokenizer.encode(final_prediction, add_special_tokens=False)
        if not final_token_ids:
            return {}
        final_token_id = final_token_ids[0]
        
        # Track probability of final prediction across layers
        convergence_probs = []
        convergence_layer = None
        
        for layer_idx in layers:
            if layer_idx not in result.layer_probabilities:
                continue
            
            probs = result.layer_probabilities[layer_idx]
            
            if position >= probs.shape[0]:
                continue
            
            # Get probability of final prediction
            final_prob = probs[position, final_token_id].item()
            convergence_probs.append(final_prob)
            
            # Check if this is where prediction converges (>50% probability)
            if convergence_layer is None and final_prob > 0.5:
                convergence_layer = layer_idx
        
        return {
            "convergence_layer": convergence_layer,
            "final_prediction": final_prediction,
            "convergence_probabilities": convergence_probs,
            "layers_analyzed": layers
        }