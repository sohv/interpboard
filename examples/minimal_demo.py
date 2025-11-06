"""
minimal quick start script demonstrating the basic structure.
"""
import sys
import os

# add the parent directory to the path so we can import interpboard
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import interpboard
import torch
import transformers
import matplotlib
import numpy
from interpboard import attribution, patching, visualization, circuits

def main():    
    print(" InterpBoard - Quick Start")
    print("=" * 50)

    print(" testing package imports...")

    # Core module already imported at the top
    print(" interpboard imported successfully")
    print(f" Package version: {getattr(interpboard, '__version__', 'Unknown')}")

    # Since optional dependencies are imported at the top, mark them as available
    deps = {
        'torch': True,
        'transformers': True,
        'matplotlib': True,
        'numpy': True,
        'all_available': True
    }

    print(f"Dependency Status:")
    print(f"  PyTorch: {'YES' if deps['torch'] else 'NO'}")
    print(f"  Transformers: {'YES' if deps['transformers'] else 'NO'}")
    print(f"  Matplotlib: {'YES' if deps['matplotlib'] else 'NO'}")
    print(f"  NumPy: {'YES' if deps['numpy'] else 'NO'}")

    missing = [k for k, v in deps.items() if not v and k != 'all_available']

    if missing:
        print(f"\nWARNING: Missing dependencies: {missing}")
        print("Install them with:")
        print(f"  pip install {' '.join(missing)}")
    else:
        print("\nAll dependencies available!")

    # Test basic model loading if transformers is available
    if deps['transformers'] and deps['torch']:
        print("\nTesting model loading...")
        model, tokenizer = interpboard.load_model_and_tokenizer("gpt2")
        print("Model loading successful")
    else:
        print("\nWARNING: Cannot test model loading without torch and transformers")

    # Show available components based on dependencies
    print("\n Available Components:")

    if deps['all_available']:
        print("  Attribution Methods:")
        print("    • GradientAttributor - Gradient-based attribution")
        print("    • AttentionAttributor - Attention-based attribution")
        print("    • AttributionVisualizer - Rich visualizations")

        print("  Activation Patching:")
        print("    • ActivationPatcher - Systematic ablation")
        print("    • CausalTracer - Causal tracing experiments")

        print("  Mechanistic Analysis:")
        print("    • LogitLens - Layer-by-layer predictions")
        print("    • NeuronAnalyzer - Individual neuron analysis")
        print("    • AttentionHeadAblator - Head ablation studies")

        print("  Visualization:")
        print("    • TextOverlayVisualizer - Interactive text displays")
        print("    • HeatmapVisualizer - Attention and activation heatmaps")

        print("  High-Level Dashboards:")
        print("    • AttributionDashboard - Streamlined attribution analysis")
        print("    • AblationDashboard - Comprehensive ablation tools")

        print("\n example usage:")
        print("""
        # Basic usage pattern:
        from interpboard.dashboards import create_unified_dashboard

        # Create dashboards for any model
        attribution_dashboard, ablation_dashboard = create_unified_dashboard("gpt2")

        # Run analysis
        result = attribution_dashboard.analyze(
            "The Eiffel Tower is in Paris.",
            methods=["integrated_gradients"],
            visualize=True
        )
        """)

    else:
        print("  Package structure loaded with fallback implementations")
        print("  Install missing dependencies for full functionality")

    print("\nNext Steps:")
    if missing:
        print(f"  1. Install dependencies: pip install {' '.join(missing)}")
        print("  2. Or install all at once: pip install -e .")
    else:
        print("  1. Try the full example: python examples/quick_start.py")
        print("  2. Run the GPT-2 demo: python gpt2_demo.py")
    print("  3. Explore Jupyter notebooks in examples/")
    print("  4. Read documentation in docs/")

    print("\nQuick start complete! The package structure is ready.")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)