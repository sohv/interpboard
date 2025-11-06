"""
Dashboard interfaces for easy access to interpretability tools.

This module provides high-level dashboard classes that combine multiple
interpretability methods for streamlined analysis workflows.
"""

import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import warnings

from ..attribution import GradientAttributor, AttentionAttributor, AttributionVisualizer
from ..patching import ActivationPatcher, CausalTracer
from ..circuits import LogitLens, NeuronAnalyzer, AttentionHeadAblator
from ..visualization import TextOverlayVisualizer, HeatmapVisualizer
from ..utils import load_model_and_tokenizer
from ..config import Config


@dataclass
class DashboardResult:
    """Container for dashboard analysis results."""
    text: str
    tokens: List[str]
    attribution_results: Dict[str, Any]
    patching_results: Dict[str, Any]
    attention_results: Dict[str, Any]
    mechanistic_results: Dict[str, Any]
    visualizations: Dict[str, Any]
    metadata: Dict[str, Any]


class AttributionDashboard:
    """
    Comprehensive dashboard for attribution analysis.
    
    Combines gradient-based and attention-based attribution methods
    with rich visualizations for easy analysis.
    """
    
    def __init__(self, model, tokenizer, device="auto", config=None):
        """
        Initialize the attribution dashboard.
        
        Args:
            model: The transformer model to analyze
            tokenizer: The tokenizer for the model
            device: Device to run computations on
            config: Optional configuration object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or Config()
        
        # Detect if running in remote environment (VS Code remote, SSH, etc.)
        self.is_remote = self._detect_remote_environment()
        
        # Initialize analyzers
        self.gradient_attributor = GradientAttributor(model, tokenizer, self.device)
        self.attention_attributor = AttentionAttributor(model, tokenizer, self.device)
        self.attribution_visualizer = AttributionVisualizer()
        self.text_visualizer = TextOverlayVisualizer()
    
    def analyze(
        self,
        text: str,
        methods: List[str] = None,
        target_position: int = -1,
        visualize: bool = True,
        interactive: bool = True
    ) -> DashboardResult:
        """
        Run comprehensive attribution analysis on text.
        
        Args:
            text: Input text to analyze
            methods: Attribution methods to use
            target_position: Position to analyze (default: last token)
            visualize: Whether to create visualizations
            interactive: Whether to use interactive Plotly unified dashboard (default: True)
            
        Returns:
            DashboardResult with all analysis results
        """
        if methods is None:
            methods = ["vanilla_gradient", "integrated_gradients", "attention_rollout"]
        
        print(f"Attribution Analysis: '{text}'")
        print(f"Methods: {methods}")
        print("=" * 50)
        
        # Tokenize input
        tokens = self.tokenizer.tokenize(text)
        
        # Run gradient-based methods
        gradient_methods = [m for m in methods if m in ["vanilla_gradient", "gradient_x_input", "integrated_gradients"]]
        attribution_results = {}
        
        if gradient_methods:
            print("🧮 Computing gradient-based attributions...")
            grad_results = self.gradient_attributor.compute_all_attributions(
                text, methods=gradient_methods
            )
            attribution_results.update(grad_results)
        
        # Run attention-based methods
        attention_methods = [m for m in methods if m in ["attention_rollout", "attention_flow"]]
        
        for method in attention_methods:
            print(f"Computing {method}...")
            if method == "attention_rollout":
                result = self.attention_attributor.attention_rollout(text)
                attribution_results[method] = result
            elif method == "attention_flow":
                result = self.attention_attributor.attention_flow(text)
                attribution_results[method] = result
        
        # Create visualizations
        visualizations = {}
        if visualize and attribution_results:
            print("Creating visualizations...")
            
            # Token heatmaps for each method
            for method_name, result in attribution_results.items():
                fig = self.attribution_visualizer.plot_token_heatmap(
                    result,
                    title=f"{method_name.replace('_', ' ').title()} Attribution",
                    show_values=True
                )
                visualizations[f"{method_name}_heatmap"] = fig
                
                if visualize:
                    plt.show()
            
            # Comparison plot
            if len(attribution_results) > 1:
                comparison_fig = self.attribution_visualizer.compare_attributions(
                    attribution_results,
                    title="Attribution Method Comparison"
                )
                visualizations["comparison"] = comparison_fig
                
                if visualize:
                    plt.show()
            
            # Unified Interactive Dashboard (default)
            if interactive and attribution_results:
                print("Creating unified interactive dashboard...")
                unified_fig = self._create_unified_interactive_dashboard(
                    attribution_results, 
                    text,
                    title="InterpBoard Analysis Dashboard"
                )
                
                # Handle display based on environment
                if self.is_remote:
                    # Save unified dashboard as single HTML file
                    html_path = "interpboard_dashboard.html"
                    unified_fig.write_html(html_path)
                    print(f"Saved unified interactive dashboard to {html_path}")
                    print(f"Open {html_path} in your browser to view all analyses")
                else:
                    # Show directly for local environments
                    print(f"Displaying unified interactive dashboard...")
                    unified_fig.show()
                    
                visualizations["unified_dashboard"] = unified_fig
        
        # Analyze results
        self._print_attribution_summary(attribution_results)
        
        return DashboardResult(
            text=text,
            tokens=tokens,
            attribution_results=attribution_results,
            patching_results={},
            attention_results={},
            mechanistic_results={},
            visualizations=visualizations,
            metadata={
                "methods_used": methods,
                "target_position": target_position,
                "device": self.device
            }
        )
    
    def compare_texts(
        self,
        texts: List[str],
        method: str = "integrated_gradients",
        visualize: bool = True,
        interactive: bool = True
    ) -> Dict[str, DashboardResult]:
        """
        Compare attribution patterns across multiple texts.
        
        Args:
            texts: List of texts to compare
            method: Attribution method to use
            visualize: Whether to create comparison visualizations
            interactive: Whether to use interactive Plotly visualizations
            
        Returns:
            Dictionary mapping text to analysis results
        """
        print(f"Comparative Attribution Analysis")
        print(f"Method: {method}")
        print(f"Texts: {len(texts)}")
        print("=" * 50)
        
        results = {}
        
        for i, text in enumerate(texts, 1):
            print(f"\n{i}. Analyzing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            result = self.analyze(
                text,
                methods=[method],
                visualize=False
            )
            results[text] = result
        
        if visualize:
            if interactive:
                self._create_interactive_comparison_visualization(results, method)
            else:
                self._create_comparison_visualization(results, method)
        
        return results
    
    def _create_unified_interactive_dashboard(self, attribution_results, text, title="InterpBoard Dashboard"):
        """Create a unified interactive dashboard with all analysis results."""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        num_methods = len(attribution_results)
        
        # Create subplots with proper spacing to avoid text overlap
        fig = make_subplots(
            rows=num_methods,
            cols=1,
            subplot_titles=[f"{method.replace('_', ' ').title()} Attribution" 
                          for method in attribution_results.keys()],
            vertical_spacing=0.15,  # Increased spacing to prevent overlap
            specs=[[{"secondary_y": False}] for _ in range(num_methods)]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (method_name, result) in enumerate(attribution_results.items(), 1):
            attrs = result.attributions.detach().cpu().numpy()
            tokens = result.tokens
            
            # Create interactive bar chart with hover information
            hover_template = (
                "<b>Token:</b> %{text}<br>"
                "<b>Attribution:</b> %{y:.6f}<br>"
                "<b>Position:</b> %{x}<br>"
                "<b>Method:</b> " + method_name.replace('_', ' ').title() +
                "<extra></extra>"
            )
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(tokens))),
                    y=attrs,
                    text=tokens,
                    name=method_name.replace('_', ' ').title(),
                    marker_color=colors[i % len(colors)],
                    hovertemplate=hover_template,
                    textposition='outside',
                    textangle=45,  # Angle text to prevent overlap
                    textfont=dict(size=10)  # Smaller font to prevent overlap
                ),
                row=i, col=1
            )
            
            # Update subplot axes with better formatting
            fig.update_xaxes(
                title_text="Token Position" if i == num_methods else "",
                tickmode='array',
                tickvals=list(range(0, len(tokens), max(1, len(tokens)//10))),  # Show fewer ticks
                ticktext=[tokens[j] for j in range(0, len(tokens), max(1, len(tokens)//10))],
                tickangle=45,
                tickfont=dict(size=9),
                row=i, col=1
            )
            
            fig.update_yaxes(
                title_text="Attribution Score",
                tickfont=dict(size=9),
                row=i, col=1
            )
        
        # Update overall layout with better spacing
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>Text: {text[:80]}{'...' if len(text) > 80 else ''}</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            height=400 * num_methods,  # Dynamic height based on number of methods
            width=1400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(size=11),
            margin=dict(t=120, b=80, l=80, r=80)  # Better margins
        )
        
        return fig
    
    def _detect_remote_environment(self) -> bool:
        """Detect if running in a remote environment (VS Code remote, SSH, Docker, etc.)."""
        import os
        
        # Check for common remote environment indicators
        remote_indicators = [
            os.environ.get('SSH_CONNECTION'),
            os.environ.get('SSH_CLIENT'), 
            os.environ.get('VSCODE_IPC_HOOK'),
            os.environ.get('REMOTE_CONTAINERS'),
            os.environ.get('CODESPACES')
        ]
        
        return any(indicator for indicator in remote_indicators)
    
    def _print_attribution_summary(self, attribution_results: Dict[str, Any]):
        """Print summary of attribution results."""
        print("\nATTRIBUTION SUMMARY:")
        
        for method_name, result in attribution_results.items():
            print(f"\n{method_name.upper()}:")
            
            # Find top attributed tokens
            top_indices = torch.argsort(result.attributions, descending=True)[:5]
            print("  Top attributed tokens:")
            
            for i, idx in enumerate(top_indices):
                token = result.tokens[idx]
                score = result.attributions[idx].item()
                print(f"    {i+1}. '{token}': {score:.4f}")
            
            # Print statistics
            attrs = result.attributions
            print(f"  Statistics:")
            print(f"    Mean: {attrs.mean().item():.4f}")
            print(f"    Std: {attrs.std().item():.4f}")
            print(f"    Max: {attrs.max().item():.4f}")
            print(f"    Min: {attrs.min().item():.4f}")
    
    def _create_comparison_visualization(self, results: Dict[str, DashboardResult], method: str):
        """Create visualization comparing attribution patterns across texts."""
        print(f"\nCreating comparison visualization...")
        
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 3 * len(results)))
        if len(results) == 1:
            axes = [axes]
        
        for i, (text, result) in enumerate(results.items()):
            attribution_result = result.attribution_results[method]
            
            # Plot attribution heatmap
            attrs = attribution_result.attributions.detach().cpu().numpy()
            tokens = attribution_result.tokens
            
            im = axes[i].imshow(attrs.reshape(1, -1), cmap='RdBu_r', aspect='auto')
            axes[i].set_xticks(range(len(tokens)))
            axes[i].set_xticklabels(tokens, rotation=45, ha='right')
            axes[i].set_yticks([])
            axes[i].set_title(f"Text {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
        
        plt.suptitle(f"Attribution Comparison - {method.title()}")
        plt.tight_layout()
        plt.show()
    
    def _create_interactive_comparison_visualization(self, results: Dict[str, DashboardResult], method: str):
        """Create interactive Plotly comparison visualization."""
        print("Creating interactive comparison visualization...")
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        num_texts = len(results)
        fig = make_subplots(
            rows=num_texts, cols=1,
            subplot_titles=[f"Text {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}" 
                          for i, text in enumerate(results.keys())],
            vertical_spacing=0.05
        )
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (text, result) in enumerate(results.items()):
            attribution_result = result.attribution_results[method]
            attrs = attribution_result.attributions.detach().cpu().numpy()
            tokens = attribution_result.tokens
            
            # Create bar chart for this text with improved text handling
            fig.add_trace(
                go.Bar(
                    x=list(range(len(tokens))),
                    y=attrs,
                    text=tokens,
                    textposition='outside',
                    textangle=45,  # Angle text to prevent overlap
                    textfont=dict(size=9),  # Smaller font to prevent overlap
                    name=f"Text {i+1}",
                    marker_color=colors[i % len(colors)],
                    hovertemplate="<b>Token:</b> %{text}<br><b>Attribution:</b> %{y:.6f}<br><b>Position:</b> %{x}<extra></extra>"
                ),
                row=i+1, col=1
            )
            
            # Update x-axis for this subplot with better tick handling
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(0, len(tokens), max(1, len(tokens)//8))),  # Show fewer ticks
                ticktext=[tokens[j] for j in range(0, len(tokens), max(1, len(tokens)//8))],
                tickangle=45,
                tickfont=dict(size=9),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title=f"Interactive Attribution Comparison - {method.title()}",
            height=350 * num_texts,  # Increased height for better spacing
            width=1400,
            showlegend=True,
            font=dict(size=11),
            margin=dict(t=100, b=100, l=80, r=80)  # Better margins
        )
        
        # Handle display based on environment
        if self.is_remote:
            # Save as HTML for remote environments
            html_path = "interpboard_comparison_dashboard.html" 
            fig.write_html(html_path)
            print(f"Saved interactive comparison dashboard to {html_path}")
            print(f"Open {html_path} in your browser to view the comparison analysis")
            print("Interactive comparison dashboard saved!")
        else:
            # Show directly for local environments
            fig.show()
            print("Interactive comparison dashboard displayed!")


class AblationDashboard:
    """
    Comprehensive dashboard for ablation and causal analysis.
    
    Combines activation patching, causal tracing, and mechanistic
    analysis tools for understanding model behavior.
    """
    
    def __init__(self, model, tokenizer, device="auto", config=None):
        """
        Initialize the ablation dashboard.
        
        Args:
            model: The transformer model to analyze
            tokenizer: The tokenizer for the model
            device: Device to run computations on
            config: Optional configuration object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or Config()
        
        # Initialize analyzers
        self.logit_lens = LogitLens(model, tokenizer, self.device)
        self.neuron_analyzer = NeuronAnalyzer(model, tokenizer, self.device)
        self.heatmap_visualizer = HeatmapVisualizer()
    
    def analyze(
        self,
        text: str,
        analyses: List[str] = None,
        layers: List[int] = None,
        visualize: bool = True
    ) -> DashboardResult:
        """
        Run comprehensive ablation analysis on text.
        
        Args:
            text: Input text to analyze
            analyses: Types of analysis to run
            layers: Specific layers to analyze
            visualize: Whether to create visualizations
            
        Returns:
            DashboardResult with all analysis results
        """
        if analyses is None:
            analyses = ["patching", "causal_tracing", "logit_lens", "head_ablation"]
        
        if layers is None:
            # Default to analyzing later layers
            num_layers = len(self.model.transformer.h)
            layers = list(range(max(0, num_layers - 6), num_layers))
        
        print(f"Ablation Analysis: '{text}'")
        print(f"Analyses: {analyses}")
        print(f"Layers: {layers}")
        print("=" * 50)
        
        tokens = self.tokenizer.tokenize(text)
        
        patching_results = {}
        mechanistic_results = {}
        visualizations = {}
        
        # Activation patching analysis
        if "patching" in analyses:
            print("Running activation patching...")
            try:
                with ActivationPatcher(self.model, self.tokenizer, self.device) as patcher:
                    # Find critical components
                    critical_components = patcher.find_critical_components(
                        text,
                        threshold=0.05
                    )
                    patching_results["critical_components"] = critical_components
                    
                    # Systematic ablation on subset of layers
                    if len(layers) <= 6:  # Limit to avoid too many experiments
                        ablation_results = patcher.run_systematic_ablation(
                            text,
                            layers=layers,
                            components=["attention", "mlp"],
                            heads=list(range(8))  # First 8 heads
                        )
                        patching_results["systematic_ablation"] = ablation_results
            except Exception as e:
                print(f"Activation patching failed: {e}")
                patching_results["error"] = str(e)
        
        # Causal tracing
        if "causal_tracing" in analyses:
            print("Running causal tracing...")
            try:
                with CausalTracer(self.model, self.tokenizer, self.device) as tracer:
                    # Try to identify subject tokens automatically
                    words = text.split()
                    if len(words) >= 2:
                        subject_tokens = words[:2]  # Take first two words as subject
                        trace_result = tracer.trace_causal_effect(
                            text=text,
                            subject_tokens=subject_tokens,
                            target_token_position=-1
                        )
                        patching_results["causal_tracing"] = trace_result
            except Exception as e:
                print(f"Causal tracing failed: {e}")
                patching_results["causal_tracing_error"] = str(e)
        
        # Logit lens analysis
        if "logit_lens" in analyses:
            print("Running logit lens analysis...")
            lens_result = self.logit_lens.analyze(
                text,
                layers=layers,
                top_k=5
            )
            mechanistic_results["logit_lens"] = lens_result
            
            # Analyze convergence
            last_position = len(tokens) - 1
            convergence = self.logit_lens.analyze_convergence(
                lens_result,
                position=last_position
            )
            mechanistic_results["convergence"] = convergence
        
        # Head ablation analysis
        if "head_ablation" in analyses:
            print("Running head ablation analysis...")
            try:
                with AttentionHeadAblator(self.model, self.tokenizer, self.device) as ablator:
                    # Focus on subset of layers and heads to avoid too many experiments
                    test_layers = layers[-3:] if len(layers) > 3 else layers
                    test_heads = list(range(8))  # First 8 heads
                    
                    head_results = ablator.systematic_head_ablation(
                        text,
                        layers=test_layers,
                        heads=test_heads,
                        ablation_type="zero"
                    )
                    mechanistic_results["head_ablation"] = head_results
                    
                    # Find critical heads
                    critical_heads = ablator.find_critical_heads(
                        head_results,
                        threshold=0.05,
                        top_k=10
                    )
                    mechanistic_results["critical_heads"] = critical_heads
            except Exception as e:
                print(f"Head ablation failed: {e}")
                mechanistic_results["head_ablation_error"] = str(e)
        
        # Create visualizations
        if visualize:
            print("Creating visualizations...")
            visualizations = self._create_visualizations(
                patching_results, mechanistic_results, text, layers
            )
        
        # Print summary
        self._print_ablation_summary(patching_results, mechanistic_results)
        
        return DashboardResult(
            text=text,
            tokens=tokens,
            attribution_results={},
            patching_results=patching_results,
            attention_results={},
            mechanistic_results=mechanistic_results,
            visualizations=visualizations,
            metadata={
                "analyses_used": analyses,
                "layers_analyzed": layers,
                "device": self.device
            }
        )
    
    def _create_visualizations(
        self,
        patching_results: Dict[str, Any],
        mechanistic_results: Dict[str, Any],
        text: str,
        layers: List[int]
    ) -> Dict[str, Any]:
        """Create visualizations for ablation results."""
        visualizations = {}
        
        # Convergence plot
        if "convergence" in mechanistic_results:
            convergence = mechanistic_results["convergence"]
            if "convergence_probabilities" in convergence:
                probs = convergence["convergence_probabilities"]
                conv_layers = convergence["layers_analyzed"]
                
                plt.figure(figsize=(10, 6))
                plt.plot(conv_layers, probs, 'o-', linewidth=2, markersize=8)
                plt.xlabel('Layer')
                plt.ylabel('Probability of Final Prediction')
                plt.title('Prediction Convergence Across Layers')
                plt.grid(True, alpha=0.3)
                visualizations["convergence_plot"] = plt.gcf()
                plt.show()
        
        # Head ablation heatmap
        if "head_ablation" in mechanistic_results:
            with AttentionHeadAblator(self.model, self.tokenizer, self.device) as ablator:
                head_results = mechanistic_results["head_ablation"]
                heatmap_matrix = ablator.layer_head_heatmap(head_results)
                
                plt.figure(figsize=(12, 8))
                plt.imshow(heatmap_matrix, cmap='Reds', aspect='auto')
                plt.colorbar(label='Ablation Impact')
                plt.xlabel('Head Index')
                plt.ylabel('Layer Index')
                plt.title('Attention Head Ablation Impact')
                
                # Add annotations for high impact
                for i in range(heatmap_matrix.shape[0]):
                    for j in range(heatmap_matrix.shape[1]):
                        if heatmap_matrix[i, j] > 0.1:
                            plt.text(j, i, f'{heatmap_matrix[i, j]:.2f}', 
                                   ha='center', va='center', color='white')
                
                visualizations["head_ablation_heatmap"] = plt.gcf()
                plt.tight_layout()
                plt.show()
        
        return visualizations
    
    def _print_ablation_summary(
        self,
        patching_results: Dict[str, Any],
        mechanistic_results: Dict[str, Any]
    ):
        """Print summary of ablation results."""
        print("\nABLATION SUMMARY:")
        
        # Critical components
        if "critical_components" in patching_results:
            critical = patching_results["critical_components"]
            print(f"\n  Found {len(critical)} critical components:")
            for location, impact in critical[:5]:
                comp_name = f"Layer {location.layer_idx}, {location.component}"
                if location.head_idx is not None:
                    comp_name += f", Head {location.head_idx}"
                print(f"    {comp_name}: {impact:.4f}")
        
        # Causal tracing
        if "causal_tracing" in patching_results:
            trace = patching_results["causal_tracing"]
            print(f"\n  Causal tracing:")
            print(f"    Baseline score: {trace.baseline_score:.4f}")
            print(f"    Corrupted score: {trace.corrupted_score:.4f}")
            print(f"    Subject positions: {trace.subject_token_positions}")
        
        # Logit lens
        if "logit_lens" in mechanistic_results:
            lens = mechanistic_results["logit_lens"]
            print(f"\n  Logit lens:")
            print(f"    Analyzed {len(lens.layer_predictions)} layers")
            
            if "convergence" in mechanistic_results:
                conv = mechanistic_results["convergence"]
                print(f"    Final prediction: {conv.get('final_prediction', 'N/A')}")
                print(f"    Convergence layer: {conv.get('convergence_layer', 'N/A')}")
        
        # Head ablation
        if "critical_heads" in mechanistic_results:
            critical_heads = mechanistic_results["critical_heads"]
            print(f"\n  Found {len(critical_heads)} critical attention heads:")
            for head_id, impact in critical_heads[:5]:
                print(f"    {head_id}: {impact:.4f}")


def create_unified_dashboard(
    model_name: str = "gpt2",
    device: str = "auto",
    config: Optional[Config] = None
) -> tuple[AttributionDashboard, AblationDashboard]:
    """
    Create unified dashboard with both attribution and ablation tools.
    
    Args:
        model_name: Name or path of the model to load
        device: Device to run computations on
        config: Optional configuration object
        
    Returns:
        Tuple of (AttributionDashboard, AblationDashboard)
    """
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=device if device != "auto" else None
    )
    
    actual_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dashboards
    attribution_dashboard = AttributionDashboard(model, tokenizer, actual_device, config)
    ablation_dashboard = AblationDashboard(model, tokenizer, actual_device, config)
    
    print(f"Dashboards ready! Device: {actual_device}")
    
    return attribution_dashboard, ablation_dashboard