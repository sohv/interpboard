"""
Minimal quick start script demonstrating the basic structure.

This simplified version shows the library structure without requiring
full installation of all dependencies.
"""

import sys
import os

# Add the parent directory to the path so we can import interpboard
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    """Run minimal quick start example."""
    
    print("🚀 InterpBoard - Quick Start")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("📦 Testing package imports...")
        
        # Test core modules
        import interpboard
        print("✅ interpboard imported successfully")
        
        print(f"📊 Package version: {getattr(interpboard, '__version__', 'Unknown')}")
        
        # Test dependency availability by trying imports
        deps = {
            'torch': False,
            'transformers': False,
            'matplotlib': False,
            'numpy': False,
            'all_available': False
        }
        
        try:
            import torch
            deps['torch'] = True
        except ImportError:
            pass
            
        try:
            import transformers
            deps['transformers'] = True
        except ImportError:
            pass
            
        try:
            import matplotlib
            deps['matplotlib'] = True
        except ImportError:
            pass
            
        try:
            import numpy
            deps['numpy'] = True
        except ImportError:
            pass
            
        deps['all_available'] = all([deps['torch'], deps['transformers'], deps['matplotlib'], deps['numpy']])
        
        print(f"📊 Dependency Status:")
        print(f"  PyTorch: {'✅' if deps['torch'] else '❌'}")
        print(f"  Transformers: {'✅' if deps['transformers'] else '❌'}")
        print(f"  Matplotlib: {'✅' if deps['matplotlib'] else '❌'}")
        print(f"  NumPy: {'✅' if deps['numpy'] else '❌'}")
        
        missing = [k for k, v in deps.items() if not v and k != 'all_available']
        
        if missing:
            print(f"\n⚠️  Missing dependencies: {missing}")
            print("Install them with:")
            print(f"  pip install {' '.join(missing)}")
        else:
            print("\n✅ All dependencies available!")
        
        # Test basic model loading if transformers is available
        if deps['transformers'] and deps['torch']:
            print("\n🔧 Testing model loading...")
            model, tokenizer = interpboard.load_model_and_tokenizer("gpt2")
            print("✅ Model loading successful")
        else:
            print("\n⚠️  Cannot test model loading without torch and transformers")
        
        # Show available components based on dependencies
        print("\n🧰 Available Components:")
        
        if deps['all_available']:
            try:
                from interpboard import attribution, patching, visualization, circuits
                print("  📈 Attribution Methods:")
                print("    • GradientAttributor - Gradient-based attribution")
                print("    • AttentionAttributor - Attention-based attribution")
                print("    • AttributionVisualizer - Rich visualizations")
                
                print("  🔧 Activation Patching:")
                print("    • ActivationPatcher - Systematic ablation")
                print("    • CausalTracer - Causal tracing experiments")
                
                print("  🔬 Mechanistic Analysis:")
                print("    • LogitLens - Layer-by-layer predictions")
                print("    • NeuronAnalyzer - Individual neuron analysis")
                print("    • AttentionHeadAblator - Head ablation studies")
                
                print("  📊 Visualization:")
                print("    • TextOverlayVisualizer - Interactive text displays")
                print("    • HeatmapVisualizer - Attention and activation heatmaps")
                
                print("  🎛️ High-Level Dashboards:")
                print("    • AttributionDashboard - Streamlined attribution analysis")
                print("    • AblationDashboard - Comprehensive ablation tools")
                
                print("\n🎯 Example Usage:")
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
                
            except ImportError as ie:
                print(f"  ⚠️  Some modules not available: {ie}")
                print("  📦 Core structure is available but functionality is limited")
        
        else:
            print("  📦 Package structure loaded with fallback implementations")
            print("  🔧 Install missing dependencies for full functionality")
        
        print("\n📚 Next Steps:")
        if missing:
            print(f"  1. Install dependencies: pip install {' '.join(missing)}")
            print("  2. Or install all at once: pip install -e .")
        else:
            print("  1. Try the full example: python examples/quick_start.py")
            print("  2. Run the GPT-2 demo: python gpt2_demo.py")
        print("  3. Explore Jupyter notebooks in examples/")
        print("  4. Read documentation in docs/")
        
        print("\n✅ Quick start complete! The package structure is ready.")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Make sure you're in the correct directory")
        print("  2. Check that all files exist in interpboard/")
        print("  3. Install missing dependencies")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)