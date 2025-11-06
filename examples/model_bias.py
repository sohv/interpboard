import sys
import os

# add the parent directory to the path so we can import interpboard
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interpboard.dashboards import create_unified_dashboard

# Analyze gender bias in language models
attribution_dashboard, _ = create_unified_dashboard("gpt2")

bias_texts = [
    "The doctor walked into the room and he",
    "The doctor walked into the room and she", 
    "The nurse prepared the medication and he",
    "The nurse prepared the medication and she"
]

# Compare attribution patterns to identify bias
comparison_results = attribution_dashboard.compare_texts(
    bias_texts,
    method="integrated_gradients",
    interactive=True
)