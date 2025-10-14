"""
Attribution module for gradient and attention-based interpretability methods.
"""

from .gradients import (
    GradientAttributor,
    AttributionResult
)

from .attention_rollout import (
    AttentionAttributor
)

from .viz import (
    AttributionVisualizer
)

__all__ = [
    # Core attribution
    "GradientAttributor",
    "AttributionResult",
    "AttentionAttributor",
    
    # Visualization
    "AttributionVisualizer"
]