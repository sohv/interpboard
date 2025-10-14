"""
Patching module for activation interventions and causal analysis.
"""

from .activ_patch import (
    ActivationPatcher,
    PatchLocation,
    PatchType,
    PatchResult
)

from .causal_tracing import (
    CausalTracer,
    CausalTraceConfig,
    CausalTraceResult
)

from .utils import (
    find_module_by_name,
    get_activation_statistics,
    compute_activation_similarity,
    create_noise_injection_hook,
    create_zeroing_hook,
    analyze_patch_impact_distribution,
    rank_components_by_impact
)

__all__ = [
    # Core patching
    "ActivationPatcher",
    "PatchLocation", 
    "PatchType",
    "PatchResult",
    
    # Causal tracing
    "CausalTracer",
    "CausalTraceConfig",
    "CausalTraceResult",
    
    # Utilities
    "find_module_by_name",
    "get_activation_statistics",
    "compute_activation_similarity",
    "create_noise_injection_hook",
    "create_zeroing_hook",
    "analyze_patch_impact_distribution",
    "rank_components_by_impact"
]