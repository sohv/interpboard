"""
Quick start script demonstrating InterpBoard.
This shows how to use the high-level dashboard interfaces for rapid interpretability analysis.
"""

import sys
import os

# Add the parent directory to the path so we can import interpboard
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from interpboard.dashboards import create_unified_dashboard
    from interpboard.utils import load_model_and_tokenizer
except ImportError as e:
    print(" import error: Please install the package first:")
    print("   cd /path/to/interpboard")
    print("   pip install -e .")
    print(f"   Error details: {e}")
    sys.exit(1)

def main():
    """Run quick start example with dashboards."""
    
    # Create unified dashboards
    print(" creating InterpBoard Dashboards...")
    attribution_dashboard, ablation_dashboard = create_unified_dashboard(
        model_name="gpt2",
        device="auto"
    )
    
    # Example text for analysis
    text = "The Eiffel Tower is located in Paris, France."
    
    print(f"\n Analyzing: '{text}'")
    
    # Attribution analysis
    print("\n" + "="*60)
    print(" Attribution analysis")
    print("="*60)
    
    attribution_result = attribution_dashboard.analyze(
        text,
        methods=["integrated_gradients", "attention_rollout"],
        visualize=True,
        interactive=True
    )
    
    # Ablation analysis
    print("\n" + "="*60)
    print(" Ablation analysis")
    print("="*60)
    
    ablation_result = ablation_dashboard.analyze(
        text,
        analyses=["patching", "logit_lens", "head_ablation"],
        layers=[8, 9, 10, 11],
        visualize=True
    )
    
    # quick comparison across different texts
    print("\n" + "="*60)
    print(" Comparative Analysis")
    print("="*60)
    
    test_texts = [
        "The Eiffel Tower is located in Paris, France.",
        "Albert Einstein developed the theory of relativity.",
        "The Amazon River flows through Brazil."
    ]
    
    comparison_results = attribution_dashboard.compare_texts(
        test_texts,
        method="integrated_gradients",
        visualize=True
    )
    
    print("\n Analysis complete!")
    print("\nKey insights:")
    
    # Extract and display key insights
    if attribution_result.attribution_results:
        ig_result = attribution_result.attribution_results.get("integrated_gradients")
        if ig_result:
            top_idx = ig_result.attributions.argmax()
            top_token = ig_result.tokens[top_idx]
            top_score = ig_result.attributions[top_idx].item()
            print(f"  • Most important token: '{top_token}' (score: {top_score:.3f})")
    
    if ablation_result.patching_results.get("critical_components"):
        critical = ablation_result.patching_results["critical_components"]
        if critical:
            location, impact = critical[0]
            comp_name = f"Layer {location.layer_idx}, {location.component}"
            if location.head_idx is not None:
                comp_name += f", Head {location.head_idx}"
            print(f"  • Most critical component: {comp_name} (impact: {impact:.3f})")
    
    if ablation_result.mechanistic_results.get("critical_heads"):
        critical_heads = ablation_result.mechanistic_results["critical_heads"]
        if critical_heads:
            head_id, impact = critical_heads[0]
            print(f"  • Most critical attention head: {head_id} (impact: {impact:.3f})")
    
    print(f"\n Results saved in:")
    print(f"  • Interactive HTML: attribution_integrated_gradients.html")
    print(f"  • Matplotlib figures: displayed above")
    
    print(f"\n Next steps:")
    print(f"  • Try with different models (GPT-2 medium/large, GPT-Neo)")
    print(f"  • Analyze more complex texts and tasks")
    print(f"  • Explore the full API in the example notebooks")


if __name__ == "__main__":
    main()