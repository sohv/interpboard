"""
Neuron analysis for understanding individual neuron activations and behaviors.

This module provides tools to analyze activation patterns, find maximally
activating inputs, and understand what individual neurons represent.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from dataclasses import dataclass
import logging
from collections import defaultdict

from ..utils import tokenize_input, get_model_info, batch_process

logger = logging.getLogger(__name__)


@dataclass
class NeuronActivationResult:
    """Results from neuron activation analysis."""
    neuron_activations: Dict[str, torch.Tensor]  # {layer_component: [seq_len, num_neurons]}
    max_activating_tokens: Dict[str, List[Tuple[str, float]]]  # {neuron_id: [(token, activation)]}
    activation_statistics: Dict[str, Dict[str, float]]  # {neuron_id: {stat: value}}
    input_text: str
    tokens: List[str]
    metadata: Dict


@dataclass
class NeuronProbe:
    """Information about a specific neuron."""
    layer_idx: int
    component: str  # "mlp", "attention"
    neuron_idx: int
    mean_activation: float
    std_activation: float
    max_activation: float
    sparsity: float  # Fraction of zero/near-zero activations


class NeuronAnalyzer:
    """Analyzes individual neuron behaviors in transformer models."""
    
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
        
        # Cache for neuron activations
        self.activation_cache = defaultdict(list)
        
    def extract_neuron_activations(
        self,
        text: str,
        components: Optional[List[str]] = None,
        layers: Optional[List[int]] = None
    ) -> NeuronActivationResult:
        """
        Extract activations from all neurons in specified components.
        
        Args:
            text: Input text to analyze
            components: Components to analyze ("mlp", "attention")
            layers: Specific layers to analyze
            
        Returns:
            NeuronActivationResult with activation data
        """
        if components is None:
            components = ["mlp"]  # Focus on MLP neurons by default
        
        if layers is None:
            layers = list(range(self.model_info["num_layers"]))
        
        inputs = tokenize_input(text, self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Store activations
        neuron_activations = {}
        activation_hooks = []
        
        def make_activation_hook(component_name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output
                
                # Store activation
                neuron_activations[component_name] = activation.detach().cpu()
            return hook
        
        # Register hooks for specified components
        for layer_idx in layers:
            for component in components:
                try:
                    module = self._get_component_module(layer_idx, component)
                    component_name = f"L{layer_idx}_{component}"
                    
                    hook = module.register_forward_hook(make_activation_hook(component_name))
                    activation_hooks.append(hook)
                    
                except Exception as e:
                    logger.warning(f"Failed to hook {component} in layer {layer_idx}: {e}")
                    continue
        
        try:
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
        
        finally:
            # Clean up hooks
            for hook in activation_hooks:
                hook.remove()
        
        # Analyze activations
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Find max activating tokens for each neuron
        max_activating_tokens = {}
        activation_statistics = {}
        
        for component_name, activations in neuron_activations.items():
            # Remove batch dimension
            if activations.ndim == 3:
                activations = activations[0]  # [seq_len, hidden_size]
            
            # Analyze each neuron
            for neuron_idx in range(activations.shape[1]):
                neuron_id = f"{component_name}_N{neuron_idx}"
                neuron_acts = activations[:, neuron_idx]
                
                # Find max activating positions
                top_k = min(5, len(tokens))
                top_values, top_indices = torch.topk(neuron_acts, top_k)
                
                max_tokens = []
                for i in range(top_k):
                    token_idx = top_indices[i].item()
                    activation_val = top_values[i].item()
                    if token_idx < len(tokens):
                        token = tokens[token_idx]
                        max_tokens.append((token, activation_val))
                
                max_activating_tokens[neuron_id] = max_tokens
                
                # Compute statistics
                activation_statistics[neuron_id] = {
                    "mean": neuron_acts.mean().item(),
                    "std": neuron_acts.std().item(),
                    "max": neuron_acts.max().item(),
                    "min": neuron_acts.min().item(),
                    "sparsity": (torch.abs(neuron_acts) < 0.01).float().mean().item()
                }
        
        result = NeuronActivationResult(
            neuron_activations=neuron_activations,
            max_activating_tokens=max_activating_tokens,
            activation_statistics=activation_statistics,
            input_text=text,
            tokens=tokens,
            metadata={
                "components_analyzed": components,
                "layers_analyzed": layers,
                "sequence_length": len(tokens)
            }
        )
        
        return result
    
    def _get_component_module(self, layer_idx: int, component: str) -> torch.nn.Module:
        """Get the module for a specific component."""
        model_type = self.model.config.model_type.lower()
        
        if "gpt2" in model_type:
            base_layer = self.model.transformer.h[layer_idx]
            if component == "mlp":
                return base_layer.mlp
            elif component == "attention":
                return base_layer.attn
        elif "llama" in model_type:
            base_layer = self.model.model.layers[layer_idx]
            if component == "mlp":
                return base_layer.mlp
            elif component == "attention":
                return base_layer.self_attn
        else:
            raise NotImplementedError(f"Component extraction not implemented for {model_type}")
    
    def find_maximally_activating_neurons(
        self,
        texts: List[str],
        top_k: int = 10,
        components: Optional[List[str]] = None,
        layers: Optional[List[int]] = None
    ) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Find neurons that activate most strongly across a dataset.
        
        Args:
            texts: List of input texts
            top_k: Number of top neurons to return
            components: Components to analyze
            layers: Layers to analyze
            
        Returns:
            Dictionary mapping neuron_id to [(text, activation, token)] tuples
        """
        if components is None:
            components = ["mlp"]
        
        if layers is None:
            layers = list(range(min(6, self.model_info["num_layers"])))  # Limit for efficiency
        
        # Collect activations across all texts
        neuron_max_activations = defaultdict(list)
        
        for text_idx, text in enumerate(texts):
            try:
                result = self.extract_neuron_activations(text, components, layers)
                
                for component_name, activations in result.neuron_activations.items():
                    if activations.ndim == 3:
                        activations = activations[0]
                    
                    tokens = result.tokens
                    
                    # Find max activation for each neuron in this text
                    for neuron_idx in range(activations.shape[1]):
                        neuron_id = f"{component_name}_N{neuron_idx}"
                        neuron_acts = activations[:, neuron_idx]
                        
                        max_val, max_pos = torch.max(neuron_acts, dim=0)
                        max_token = tokens[max_pos.item()] if max_pos.item() < len(tokens) else "UNK"
                        
                        neuron_max_activations[neuron_id].append((
                            text, max_val.item(), max_token
                        ))
                        
            except Exception as e:
                logger.warning(f"Failed to process text {text_idx}: {e}")
                continue
        
        # Find top-k neurons by maximum activation
        top_neurons = {}
        
        for neuron_id, activations in neuron_max_activations.items():
            # Sort by activation value
            activations.sort(key=lambda x: x[1], reverse=True)
            top_neurons[neuron_id] = activations[:top_k]
        
        # Sort neurons by their maximum activation
        sorted_neurons = sorted(
            top_neurons.items(),
            key=lambda x: max(act[1] for act in x[1]) if x[1] else 0,
            reverse=True
        )
        
        return dict(sorted_neurons[:top_k])
    
    def analyze_neuron_selectivity(
        self,
        neuron_id: str,
        texts: List[str],
        threshold: float = 0.1
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Analyze selectivity of a specific neuron.
        
        Args:
            neuron_id: Neuron identifier (e.g., "L3_mlp_N42")
            texts: List of input texts
            threshold: Activation threshold for selectivity
            
        Returns:
            Selectivity analysis results
        """
        # Parse neuron ID
        parts = neuron_id.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid neuron_id format: {neuron_id}")
        
        layer_idx = int(parts[0][1:])  # Remove 'L' prefix
        component = parts[1]
        neuron_idx = int(parts[2][1:])  # Remove 'N' prefix
        
        activating_texts = []
        all_activations = []
        
        for text in texts:
            try:
                result = self.extract_neuron_activations(
                    text, [component], [layer_idx]
                )
                
                component_name = f"L{layer_idx}_{component}"
                if component_name in result.neuron_activations:
                    activations = result.neuron_activations[component_name]
                    if activations.ndim == 3:
                        activations = activations[0]
                    
                    if neuron_idx < activations.shape[1]:
                        neuron_acts = activations[:, neuron_idx]
                        max_activation = neuron_acts.max().item()
                        all_activations.append(max_activation)
                        
                        if max_activation >= threshold:
                            activating_texts.append(text)
                            
            except Exception as e:
                logger.warning(f"Failed to analyze text for neuron {neuron_id}: {e}")
                continue
        
        # Compute selectivity metrics
        if not all_activations:
            return {"selectivity": 0.0, "activating_texts": []}
        
        all_activations = np.array(all_activations)
        
        # Selectivity = fraction of inputs that activate the neuron
        selectivity = len(activating_texts) / len(texts)
        
        return {
            "selectivity": selectivity,
            "mean_activation": all_activations.mean(),
            "std_activation": all_activations.std(),
            "max_activation": all_activations.max(),
            "activating_texts": activating_texts[:10],  # Top 10 examples
            "num_activating": len(activating_texts),
            "total_texts": len(texts)
        }
    
    def compare_neuron_patterns(
        self,
        neuron_ids: List[str],
        texts: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare activation patterns between neurons.
        
        Args:
            neuron_ids: List of neuron identifiers
            texts: List of input texts
            
        Returns:
            Correlation matrix between neurons
        """
        # Collect activation patterns for each neuron
        neuron_patterns = {}
        
        for neuron_id in neuron_ids:
            # Parse neuron ID
            parts = neuron_id.split("_")
            layer_idx = int(parts[0][1:])
            component = parts[1]
            neuron_idx = int(parts[2][1:])
            
            activations = []
            
            for text in texts:
                try:
                    result = self.extract_neuron_activations(
                        text, [component], [layer_idx]
                    )
                    
                    component_name = f"L{layer_idx}_{component}"
                    if component_name in result.neuron_activations:
                        acts = result.neuron_activations[component_name]
                        if acts.ndim == 3:
                            acts = acts[0]
                        
                        if neuron_idx < acts.shape[1]:
                            max_act = acts[:, neuron_idx].max().item()
                            activations.append(max_act)
                        else:
                            activations.append(0.0)
                    else:
                        activations.append(0.0)
                        
                except Exception:
                    activations.append(0.0)
            
            neuron_patterns[neuron_id] = np.array(activations)
        
        # Compute correlations
        correlations = {}
        
        for neuron1 in neuron_ids:
            correlations[neuron1] = {}
            pattern1 = neuron_patterns[neuron1]
            
            for neuron2 in neuron_ids:
                pattern2 = neuron_patterns[neuron2]
                
                # Compute Pearson correlation
                if len(pattern1) > 1 and len(pattern2) > 1:
                    corr = np.corrcoef(pattern1, pattern2)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                
                correlations[neuron1][neuron2] = corr
        
        return correlations
    
    def find_neuron_clusters(
        self,
        texts: List[str],
        components: Optional[List[str]] = None,
        layers: Optional[List[int]] = None,
        correlation_threshold: float = 0.7,
        min_cluster_size: int = 3
    ) -> List[List[str]]:
        """
        Find clusters of neurons with similar activation patterns.
        
        Args:
            texts: List of input texts
            components: Components to analyze
            layers: Layers to analyze
            correlation_threshold: Minimum correlation for clustering
            min_cluster_size: Minimum neurons per cluster
            
        Returns:
            List of neuron clusters (each cluster is a list of neuron IDs)
        """
        if components is None:
            components = ["mlp"]
        
        if layers is None:
            layers = list(range(min(4, self.model_info["num_layers"])))
        
        # Collect all neuron IDs
        all_neuron_ids = []
        for layer_idx in layers:
            for component in components:
                # Estimate number of neurons (this is approximate)
                if component == "mlp":
                    # MLP typically has intermediate size neurons
                    num_neurons = getattr(self.model.config, "intermediate_size", 
                                        self.model_info["hidden_size"] * 4)
                else:
                    num_neurons = self.model_info["hidden_size"]
                
                # Limit to manageable number for clustering
                num_neurons = min(num_neurons, 100)
                
                for neuron_idx in range(num_neurons):
                    neuron_id = f"L{layer_idx}_{component}_N{neuron_idx}"
                    all_neuron_ids.append(neuron_id)
        
        # Sample subset if too many neurons
        if len(all_neuron_ids) > 200:
            all_neuron_ids = all_neuron_ids[:200]
        
        # Compute correlation matrix
        logger.info(f"Computing correlations for {len(all_neuron_ids)} neurons")
        correlations = self.compare_neuron_patterns(all_neuron_ids, texts[:20])  # Limit texts for efficiency
        
        # Simple clustering based on correlation threshold
        clusters = []
        used_neurons = set()
        
        for neuron1 in all_neuron_ids:
            if neuron1 in used_neurons:
                continue
                
            cluster = [neuron1]
            used_neurons.add(neuron1)
            
            for neuron2 in all_neuron_ids:
                if neuron2 in used_neurons:
                    continue
                
                if (neuron1 in correlations and neuron2 in correlations[neuron1] and
                    correlations[neuron1][neuron2] >= correlation_threshold):
                    cluster.append(neuron2)
                    used_neurons.add(neuron2)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        logger.info(f"Found {len(clusters)} neuron clusters")
        return clusters
    
    def get_neuron_summary(
        self,
        neuron_id: str,
        example_texts: List[str]
    ) -> Dict[str, Union[str, float, List[str]]]:
        """
        Get comprehensive summary of a neuron's behavior.
        
        Args:
            neuron_id: Neuron identifier
            example_texts: Example texts for analysis
            
        Returns:
            Comprehensive neuron summary
        """
        # Analyze selectivity
        selectivity_result = self.analyze_neuron_selectivity(
            neuron_id, example_texts
        )
        
        # Get maximally activating examples
        max_examples = self.find_maximally_activating_neurons(
            example_texts[:10], top_k=5, 
            components=[neuron_id.split("_")[1]],
            layers=[int(neuron_id.split("_")[0][1:])]
        )
        
        neuron_examples = max_examples.get(neuron_id, [])
        
        summary = {
            "neuron_id": neuron_id,
            "selectivity": selectivity_result["selectivity"],
            "mean_activation": selectivity_result["mean_activation"],
            "max_activation": selectivity_result["max_activation"],
            "num_activating_texts": selectivity_result["num_activating"],
            "top_activating_examples": [ex[0][:100] + "..." if len(ex[0]) > 100 else ex[0] 
                                      for ex in neuron_examples],
            "activation_values": [ex[1] for ex in neuron_examples],
            "max_activating_tokens": [ex[2] for ex in neuron_examples]
        }
        
        return summary