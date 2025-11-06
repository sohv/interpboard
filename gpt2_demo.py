#!/usr/bin/env python3
"""
GPT-2 Demo using InterpBoard Unified Dashboard with Interactive Plotly Visualizations

This demo showcases comprehensive interpretability analysis using the unified dashboard
with interactive Plotly visualizations that work in both local and remote environments.
"""

import torch
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from interpboard.dashboards import create_unified_dashboard
    INTERPBOARD_AVAILABLE = True
    print("✅ InterpBoard dashboards available")
except ImportError as e:
    print(f"❌ InterpBoard not available: {e}")
    print("Please install dependencies: pip install torch transformers matplotlib seaborn plotly rich tqdm einops")
    INTERPBOARD_AVAILABLE = False

def main():
    """
    Main demo function showcasing InterpBoard unified dashboard capabilities.
    """
    print("🚀 GPT-2 InterpBoard Dashboard Demo")
    print("=" * 60)
    
    if not INTERPBOARD_AVAILABLE:
        print("❌ InterpBoard dashboards not available. Please install dependencies.")
        print("Run: pip install torch transformers matplotlib seaborn plotly rich tqdm einops")
        return False

    # Set environment for cleaner output
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    # Create unified dashboards
    print("📦 Creating unified InterpBoard dashboards...")
    try:
        attribution_dashboard, ablation_dashboard = create_unified_dashboard("gpt2")
        print("✅ Dashboards created successfully!")
        print(f"🖥️  Device: {attribution_dashboard.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in attribution_dashboard.model.parameters()):,}")
    except Exception as e:
        print(f"❌ Error creating dashboards: {e}")
        return False

    # Demo text
    demo_text = "The Eiffel Tower is located in Paris, the capital of France."
    print(f"\n🔍 Analyzing: '{demo_text}'")

    # 1. COMPREHENSIVE SINGLE TEXT ANALYSIS
    print("\n📊 COMPREHENSIVE ATTRIBUTION ANALYSIS")
    print("-" * 50)
    
    try:
        # Run all attribution methods with interactive visualizations
        result = attribution_dashboard.analyze(
            demo_text,
            methods=["vanilla_gradient", "integrated_gradients", "attention_rollout"],
            visualize=True,
            interactive=True  # Creates interactive Plotly plots
        )
        
        print(f"✅ Analysis completed for {len(result.attribution_results)} methods")
        
        # Display summary
        print("\n📈 Attribution Summary:")
        for method, method_result in result.attribution_results.items():
            attrs = method_result.attributions
            print(f"  {method.replace('_', ' ').title()}:")
            print(f"    Range: [{attrs.min():.4f}, {attrs.max():.4f}]")
            print(f"    Mean: {attrs.mean():.4f}")
            
            # Show top attributed token
            top_idx = torch.argmax(torch.abs(attrs))
            top_token = method_result.tokens[top_idx]
            top_score = attrs[top_idx].item()
            print(f"    Most important: '{top_token}' ({top_score:+.4f})")
    
    except Exception as e:
        print(f"❌ Single text analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. COMPARATIVE ANALYSIS WITH INTERACTIVE DASHBOARD
    print("\n🔄 COMPARATIVE ANALYSIS DASHBOARD")
    print("-" * 50)
    
    comparison_texts = [
        "The Eiffel Tower is located in Paris, France.",
        "Albert Einstein developed the theory of relativity.",
        "The Amazon River flows through South America.", 
        "William Shakespeare wrote Romeo and Juliet."
    ]
    
    try:
        # Create interactive comparison visualization
        comparison_results = attribution_dashboard.compare_texts(
            comparison_texts,
            method="integrated_gradients",
            visualize=True,
            interactive=True  # Creates interactive comparison plots
        )
        
        print(f"✅ Comparative analysis completed for {len(comparison_texts)} texts")
        
        # Show comparison summary
        print("\n📊 Comparison Summary:")
        for i, (text, result) in enumerate(comparison_results.items(), 1):
            method_result = result.attribution_results["integrated_gradients"]
            attrs = method_result.attributions
            
            # Find most important token
            top_idx = torch.argmax(torch.abs(attrs))
            top_token = method_result.tokens[top_idx]
            top_score = attrs[top_idx].item()
            
            text_preview = text[:40] + "..." if len(text) > 40 else text
            print(f"  {i}. {text_preview}")
            print(f"     Key token: '{top_token}' ({top_score:+.4f})")
    
    except Exception as e:
        print(f"❌ Comparative analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # 3. ABLATION ANALYSIS DEMO
    print("\n🔬 ABLATION ANALYSIS DEMO") 
    print("-" * 50)
    
    try:
        print("🔧 Running activation patching analysis...")
        
        # Simple ablation example
        patch_result = ablation_dashboard.patch_activations(
            demo_text,
            layer_range=(6, 8),  # Focus on middle layers
            visualize=True
        )
        
        print("✅ Ablation analysis completed")
        print(f"📊 Analyzed {len(patch_result.patch_effects)} patch effects")
    
    except Exception as e:
        print(f"⚠️  Ablation analysis skipped: {e}")
        print("   (This is normal - ablation requires more complex setup)")

    # 4. GENERATE SUMMARY REPORT
    print("\n📋 ANALYSIS SUMMARY REPORT")
    print("=" * 60)
    
    print(f"🎯 Demo Text: '{demo_text}'")
    print(f"🤖 Model: GPT-2 ({sum(p.numel() for p in attribution_dashboard.model.parameters()):,} params)")
    print(f"💻 Device: {attribution_dashboard.device}")
    print(f"🌍 Environment: {'Remote' if attribution_dashboard.is_remote else 'Local'}")
    
    print(f"\n📊 Generated Interactive Visualizations:")
    print(f"   • Individual attribution plots (3 methods)")  
    print(f"   • Comparative analysis plots (4 texts)")
    print(f"   • Interactive HTML files saved for viewing")
    
    print(f"\n🌐 How to View Results:")
    if attribution_dashboard.is_remote:
        print(f"   1. Interactive HTML files saved to current directory")
        print(f"   2. Download and open in your browser for full interactivity")
        print(f"   3. Files include: hover tooltips, zoom, pan, download options")
    else:
        print(f"   1. Interactive plots opened automatically in browser")  
        print(f"   2. Enjoy hover tooltips, zoom, pan, and download features")
    
    print(f"\n✅ GPT-2 InterpBoard Dashboard Demo Complete!")
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Check current directory for interactive HTML files")
    else:
        print(f"\n❌ Demo failed - check error messages above")
        sys.exit(1)