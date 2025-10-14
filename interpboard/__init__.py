"""
InterpBoard - LLM Interpretability Dashboard

A comprehensive toolkit for interpreting and analyzing transformer language models.
Provides tools for attribution analysis, activation patching, causal tracing,
and mechanistic interpretability.
"""

# Import main modules
from . import attribution
from . import patching
from . import visualization
from . import circuits
from . import dashboards

# Import core utilities
from .utils import (
    load_model_and_tokenizer,
    get_model_info,
    tokenize_input,
    extract_activations,
    compute_attention_rollout
)

from .config import (
    Config,
    ModelConfig,
    VisualizationConfig,
    PatchingConfig,
    AttributionConfig,
    CircuitConfig,
    ExperimentConfig,
    get_config,
    get_global_config,
    set_global_config
)

# Version info
__version__ = "0.1.0"
__author__ = "InterpBoard Team"

# Package-level exports
__all__ = [
    # Main modules
    "attribution",
    "patching", 
    "visualization",
    "circuits",
    "dashboards",
    
    # Utils
    "load_model_and_tokenizer",
    "get_model_info", 
    "tokenize_input",
    "extract_activations",
    "compute_attention_rollout",
    
    # Config
    "Config",
    "ModelConfig",
    "VisualizationConfig", 
    "PatchingConfig",
    "AttributionConfig",
    "CircuitConfig",
    "ExperimentConfig",
    "get_config",
    "get_global_config",
    "set_global_config",
    
    # Version
    "__version__",
    "__author__"
]