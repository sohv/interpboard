#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import seaborn as sns
import os
import sys

sys.path.append('../')

try:
    from interpboard import load_model_and_tokenizer, get_config
    from interpboard.attribution import GradientAttributor, AttentionAttributor, AttributionVisualizer
    from interpboard.patching import ActivationPatcher, CausalTracer
    from interpboard.circuits import LogitLens, NeuronAnalyzer, AttentionHeadAblator
    from interpboard.visualization import TextOverlayVisualizer, HeatmapVisualizer
    INTERPBOARD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: InterpBoard not fully available: {e}")
    print("Please install dependencies: pip install torch transformers matplotlib seaborn plotly rich tqdm einops")
    INTERPBOARD_AVAILABLE = False

def main():
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not INTERPBOARD_AVAILABLE:
        print("❌ InterpBoard not available. Please install dependencies.")
        print("Run: pip install torch transformers matplotlib seaborn plotly rich")
        return

    model_name = "gpt2"
    print(f"Loading {model_name}...")

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            local_files_only=False,
            torch_dtype=torch.float32,
            output_attentions=True,  # Force attention output
        )

        if device.type == "cuda":
            model = model.half()
        model = model.to(device)
        model.eval()

        print(f"✅ Successfully loaded model: {model_name}")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        device = torch.device("cpu")
        model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float32, output_attentions=True)
        model = model.to(device)
        model.eval()
        print(f"✅ Loaded on CPU: {model_name}")

    text = "The Eiffel Tower is located in Paris, the capital of France."
    print(f"Analyzing text: '{text}'")

    # Test if model supports attention output
    print("\n🔧 TESTING MODEL CAPABILITIES")
    try:
        test_inputs = tokenizer("Hello world", return_tensors="pt").to(device)
        with torch.no_grad():
            test_outputs = model(**test_inputs, output_attentions=True)
        
        if hasattr(test_outputs, 'attentions') and test_outputs.attentions is not None:
            if test_outputs.attentions[0] is not None:
                print("✅ Model supports attention output")
                attention_supported = True
            else:
                print("⚠️  Model returns None for attention weights")
                attention_supported = False
        else:
            print("❌ Model does not support attention output")
            attention_supported = False
    except Exception as e:
        print(f"❌ Error testing attention support: {e}")
        attention_supported = False

    # Attribution Analysis
    print("\n📊 ATTRIBUTION ANALYSIS")
    grad_attributor = GradientAttributor(model, tokenizer, device)
    
    try:
        # Try simpler gradient method first
        gradient_result = grad_attributor.vanilla_gradient(text)
        print("VANILLA GRADIENT:")
        top_indices = torch.argsort(gradient_result.attributions, descending=True)[:5]
        print("Top attributed tokens:")
        for i, idx in enumerate(top_indices):
            token = gradient_result.tokens[idx]
            score = gradient_result.attributions[idx].item()
            print(f"  {i+1}. '{token}': {score:.4f}")
    except Exception as e:
        print(f"Gradient attribution failed: {e}")

    # Attention Analysis
    print("\n👁️ ATTENTION ANALYSIS")
    if not attention_supported:
        print("⚠️  Skipping attention analysis - model does not support attention output")
    else:
        try:
            attn_attributor = AttentionAttributor(model, tokenizer, device)
            rollout_result = attn_attributor.attention_rollout(text)
            
            print("ATTENTION ROLLOUT:")
            top_indices = torch.argsort(rollout_result.attributions, descending=True)[:5]
            print("Top attributed tokens:")
            for i, idx in enumerate(top_indices):
                token = rollout_result.tokens[idx]
                score = rollout_result.attributions[idx].item()
                print(f"  {i+1}. '{token}': {score:.4f}")
        except Exception as e:
            print(f"Attention rollout failed: {e}")
            # Manual attention analysis with better error handling
            try:
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)
                
                # Check if attention weights are actually returned
                if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
                    attentions = outputs.attentions
                    # Check if first attention layer is not None
                    if attentions[0] is not None:
                        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                        
                        # Simple attention aggregation
                        last_layer_attn = attentions[-1][0]  # Last layer, remove batch dim
                        avg_attn = last_layer_attn.mean(dim=0)  # Average over heads
                        token_importance = avg_attn[-1, :]  # Attention to last token
                        
                        top_indices = torch.argsort(token_importance, descending=True)[:5]
                        print("Top attended tokens (manual calculation):")
                        for i, idx in enumerate(top_indices):
                            if idx < len(tokens):
                                token = tokens[idx]
                                score = token_importance[idx].item()
                                print(f"  {i+1}. '{token}': {score:.4f}")
                    else:
                        print("Attention layers contain None values - model may not support attention output")
                else:
                    print("Model does not return attention weights - this is normal for some model configurations")
                    print("Skipping attention analysis...")
            except Exception as e2:
                print(f"Manual attention analysis also failed: {e2}")
                print("Skipping attention analysis completely...")

    # Visualization (simplified)
    print("\n📊 VISUALIZATION")
    try:
        attr_viz = AttributionVisualizer()
        text_viz = TextOverlayVisualizer()
        
        # Try to create visualization with gradient result if available
        if 'gradient_result' in locals():
            fig = attr_viz.plot_token_heatmap(
                gradient_result,
                title="Token Attribution",
                show_values=True
            )
            plt.show()
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Skipping visualization...")

    # Activation Patching (simplified)
    print("\n🔗 ACTIVATION PATCHING")
    try:
        with ActivationPatcher(model, tokenizer, device) as patcher:
            critical_components = patcher.find_critical_components(
                text,
                metric="logit_l2_diff",
                threshold=0.1
            )
            
            print(f"Found {len(critical_components)} critical components:")
            for (location, impact), result in critical_components[:5]:
                print(f"  {location.layer_idx}_{location.component}", end="")
                if location.head_idx is not None:
                    print(f"_H{location.head_idx}", end="")
                print(f": {impact:.4f}")
    except Exception as e:
        print(f"Activation patching failed: {e}")
        print("Skipping activation patching...")

    # Causal Tracing (simplified)
    print("\n🔬 CAUSAL TRACING")
    try:
        with CausalTracer(model, tokenizer, device) as tracer:
            trace_result = tracer.trace_causal_effect(
                text="The Eiffel Tower is located in",
                subject_tokens=["Eiffel", "Tower"],
                target_token_position=-1
            )
            
            print("CAUSAL TRACING RESULTS:")
            print(f"Baseline score: {trace_result.baseline_score:.4f}")
            print(f"Corrupted score: {trace_result.corrupted_score:.4f}")
            
            sorted_effects = sorted(
                trace_result.trace_results.items(),
                key=lambda x: x[1]["restoration_effect"],
                reverse=True
            )
            
            print("Top restoration effects:")
            for component, metrics in sorted_effects[:5]:
                effect = metrics["restoration_effect"]
                print(f"  {component}: {effect:.4f}")
    except Exception as e:
        print(f"Causal tracing failed: {e}")
        print("Skipping causal tracing...")

    # Attention Visualization (manual implementation)
    print("\n👁️ ATTENTION VISUALIZATION")
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
            attentions = outputs.attentions
            # Check if attention weights are actually valid
            if attentions[0] is not None:
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                
                print(f"Got attention weights for {len(attentions)} layers")
                print(f"Tokens: {tokens}")
                
                try:
                    heatmap_viz = HeatmapVisualizer()
                    last_layer_attention = attentions[-1][0]
                    
                    fig = heatmap_viz.attention_heatmap(
                        last_layer_attention[0],
                        tokens,
                        title="Attention Pattern - Last Layer, Head 0",
                        layer_idx=len(attentions)-1,
                        head_idx=0,
                        show_values=False
                    )
                    plt.show()
                except Exception as e:
                    print(f"Attention visualization failed: {e}")
                    # Simple manual visualization
                    import matplotlib.pyplot as plt
                    last_attn = attentions[-1][0, 0].cpu().numpy()
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(last_attn, cmap='Blues')
                    plt.title("Attention Pattern (Last Layer, Head 0)")
                    plt.xlabel("Token Position")
                    plt.ylabel("Token Position")
                    plt.colorbar()
                    
                    # Add token labels
                    plt.xticks(range(len(tokens)), tokens, rotation=45)
                    plt.yticks(range(len(tokens)), tokens)
                    plt.tight_layout()
                    plt.show()
            else:
                print("Attention weights are None - model configuration issue")
        else:
            print("No attention weights available - model may not support attention output")
    except Exception as e:
        print(f"Attention visualization failed: {e}")
        print("Skipping attention visualization...")

    # Logit Lens Analysis (simplified)
    print("\n🔬 LOGIT LENS ANALYSIS")
    try:
        logit_lens = LogitLens(model, tokenizer, device)
        lens_result = logit_lens.analyze(
            text,
            layers=[0, 6, 11],
            top_k=3
        )
        
        last_position = len(lens_result.tokens) - 1
        print(f"Predictions for position {last_position}:")
        
        for layer_idx in sorted(lens_result.layer_predictions.keys()):
            if layer_idx in lens_result.top_tokens:
                top_tokens = lens_result.top_tokens[layer_idx][last_position]
                print(f"  Layer {layer_idx}: {top_tokens[:3]}")
    except Exception as e:
        print(f"Logit lens analysis failed: {e}")
        print("Skipping logit lens...")

    # Attention Head Ablation (simplified)
    print("\n🎯 ATTENTION HEAD ABLATION")
    try:
        with AttentionHeadAblator(model, tokenizer, device) as ablator:
            head_results = ablator.systematic_head_ablation(
                text,
                layers=[10, 11],
                heads=[0, 1, 2, 3],
                ablation_type="zero"
            )
            
            print("HEAD ABLATION RESULTS:")
            print(f"Baseline score: {head_results.baseline_score:.4f}")
            
            critical_heads = ablator.find_critical_heads(
                head_results,
                threshold=0.01,
                top_k=5
            )
            
            print("Most critical heads:")
            for head_id, impact in critical_heads:
                print(f"  {head_id}: {impact:.4f}")
    except Exception as e:
        print(f"Head ablation failed: {e}")
        print("Skipping head ablation...")

    # Neuron Analysis (simplified)
    print("\n🧠 NEURON ANALYSIS")
    try:
        neuron_analyzer = NeuronAnalyzer(model, tokenizer, device)
        neuron_results = neuron_analyzer.extract_neuron_activations(
            text,
            components=["mlp"],
            layers=[10]
        )
        
        print("NEURON ANALYSIS:")
        for neuron_id, stats in list(neuron_results.activation_statistics.items())[:3]:
            print(f"  {neuron_id}:")
            print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"    Max: {stats['max']:.4f}, Sparsity: {stats['sparsity']:.4f}")
    except Exception as e:
        print(f"Neuron analysis failed: {e}")
        print("Skipping neuron analysis...")

    # Comprehensive Dashboard (simplified)
    def create_interpretability_dashboard(text, model, tokenizer, device):
        print(f"\n🔍 INTERPRETABILITY DASHBOARD")
        print(f"Text: '{text}'")
        print("=" * 60)
        
        try:
            grad_attributor = GradientAttributor(model, tokenizer, device)
            ig_result = grad_attributor.vanilla_gradient(text)
            top_idx = torch.argmax(ig_result.attributions)
            top_token = ig_result.tokens[top_idx]
            top_score = ig_result.attributions[top_idx].item()
            
            print(f"\n📊 ATTRIBUTION ANALYSIS")
            print(f"Most important token: '{top_token}' (score: {top_score:.4f})")
        except Exception as e:
            print(f"\n📊 ATTRIBUTION ANALYSIS")
            print(f"Attribution analysis failed: {e}")
        
        try:
            with ActivationPatcher(model, tokenizer, device) as patcher:
                critical_components = patcher.find_critical_components(
                    text, threshold=0.05
                )
                
                print(f"\n🔗 CAUSAL ANALYSIS")
                if critical_components:
                    top_component = critical_components[0]
                    location, impact = top_component
                    comp_name = f"Layer {location.layer_idx}, {location.component}"
                    if location.head_idx is not None:
                        comp_name += f", Head {location.head_idx}"
                    print(f"Most critical component: {comp_name} (impact: {impact:.4f})")
                else:
                    print("No critical components found above threshold")
        except Exception as e:
            print(f"\n🔗 CAUSAL ANALYSIS")
            print(f"Causal analysis failed: {e}")
        
        # Simple attention analysis
        print(f"\n👁️ ATTENTION ANALYSIS")
        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            
            if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
                attentions = outputs.attentions
                # Check if attention weights are valid
                if attentions[0] is not None:
                    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                    last_layer_attn = attentions[-1][0]
                    avg_attn = last_layer_attn.mean(dim=0)
                    token_importance = avg_attn[-1, :]
                    
                    top_idx = torch.argmax(token_importance)
                    if top_idx < len(tokens):
                        top_token = tokens[top_idx]
                        top_score = token_importance[top_idx].item()
                        print(f"Most attended token: '{top_token}' (score: {top_score:.4f})")
                    else:
                        print("Attention analysis completed")
                else:
                    print("Attention weights are None - model configuration issue")
            else:
                print("No attention data available")
        except Exception as e:
            print(f"Attention analysis failed: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Dashboard complete!")

    create_interpretability_dashboard(text, model, tokenizer, device)

    # Comparative Analysis (simplified)
    test_texts = [
        "The Eiffel Tower is located in Paris, the capital of France.",
        "Albert Einstein developed the theory of relativity.", 
        "The Amazon River flows through Brazil.",
        "William Shakespeare wrote Romeo and Juliet."
    ]

    print("\n🔄 COMPARATIVE ANALYSIS")
    for i, test_text in enumerate(test_texts, 1):
        print(f"{i}. {test_text}")
        
        try:
            grad_attributor = GradientAttributor(model, tokenizer, device)
            result = grad_attributor.vanilla_gradient(test_text)
            
            top_idx = torch.argmax(result.attributions)
            top_token = result.tokens[top_idx]
            top_score = result.attributions[top_idx].item()
            
            print(f"   Most important token: '{top_token}' (score: {top_score:.4f})")
        except Exception as e:
            print(f"   Analysis failed: {e}")
        print()

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()