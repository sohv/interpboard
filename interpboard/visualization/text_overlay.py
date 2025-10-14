"""
Text overlay visualizations for interpretability results.

This module provides functions to create rich text visualizations with
color-coded backgrounds and overlays for attribution scores.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from rich.console import Console
from rich.text import Text
from rich import box
from rich.table import Table
from rich.panel import Panel
import logging

logger = logging.getLogger(__name__)


class TextOverlayVisualizer:
    """Creates text overlay visualizations for interpretability results."""
    
    def __init__(
        self,
        colormap: str = "RdBu_r",
        figsize: Tuple[int, int] = (12, 8),
        font_size: int = 12
    ):
        self.colormap = colormap
        self.figsize = figsize
        self.font_size = font_size
        self.console = Console()
    
    def create_html_overlay(
        self,
        tokens: List[str],
        scores: Union[torch.Tensor, np.ndarray],
        title: str = "Attribution Visualization",
        save_path: Optional[str] = None,
        normalize: bool = True,
        show_scores: bool = True
    ) -> str:
        """
        Create an HTML visualization with colored token backgrounds.
        
        Args:
            tokens: List of tokens
            scores: Attribution scores for each token
            title: Title for the visualization
            save_path: Path to save HTML file
            normalize: Whether to normalize scores for color mapping
            show_scores: Whether to show numerical scores
            
        Returns:
            HTML string
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        
        # Normalize scores if requested
        if normalize:
            score_min, score_max = scores.min(), scores.max()
            if score_max - score_min > 1e-8:
                norm_scores = (scores - score_min) / (score_max - score_min)
            else:
                norm_scores = np.zeros_like(scores)
        else:
            norm_scores = scores
        
        # Create HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"    <title>{title}</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .token { display: inline-block; padding: 2px 4px; margin: 1px; border-radius: 3px; }",
            "        .score { font-size: 0.8em; color: #666; }",
            "        .container { line-height: 2.5; }",
            "        h1 { color: #333; margin-bottom: 20px; }",
            "        .legend { margin-top: 20px; }",
            "        .color-bar { height: 20px; width: 300px; background: linear-gradient(to right, blue, white, red); }",
            "    </style>",
            "</head>",
            "<body>",
            f"    <h1>{title}</h1>",
            "    <div class='container'>"
        ]
        
        # Add tokens with colored backgrounds
        for token, score, norm_score in zip(tokens, scores, norm_scores):
            # Clean token for display
            display_token = token.replace('Ġ', ' ').replace('▁', ' ')
            
            # Calculate color based on normalized score
            if norm_score > 0.5:  # Positive attribution
                red = int(255 * (norm_score - 0.5) * 2)
                color = f"rgb({red}, {255 - red//2}, {255 - red//2})"
            else:  # Negative attribution
                blue = int(255 * (0.5 - norm_score) * 2)
                color = f"rgb({255 - blue//2}, {255 - blue//2}, {blue})"
            
            # Text color for contrast
            text_color = "white" if abs(norm_score - 0.5) > 0.3 else "black"
            
            # Add token span
            score_text = f" ({score:.3f})" if show_scores else ""
            html_parts.append(
                f"        <span class='token' style='background-color: {color}; color: {text_color};'>"
                f"{display_token}{score_text}</span>"
            )
        
        # Close HTML
        html_parts.extend([
            "    </div>",
            "    <div class='legend'>",
            "        <p><strong>Legend:</strong></p>",
            "        <div class='color-bar'></div>",
            "        <p>Blue: Negative Attribution | Red: Positive Attribution</p>",
            f"        <p>Score range: {scores.min():.3f} to {scores.max():.3f}</p>",
            "    </div>",
            "</body>",
            "</html>"
        ])
        
        html_string = "\n".join(html_parts)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_string)
            logger.info(f"HTML visualization saved to {save_path}")
        
        return html_string
    
    def create_plotly_overlay(
        self,
        tokens: List[str],
        scores: Union[torch.Tensor, np.ndarray],
        title: str = "Attribution Visualization",
        save_path: Optional[str] = None,
        width: int = 1200,
        height: int = 400
    ) -> go.Figure:
        """
        Create an interactive Plotly visualization.
        
        Args:
            tokens: List of tokens
            scores: Attribution scores
            title: Plot title
            save_path: Path to save HTML file
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        
        # Clean tokens for display
        display_tokens = [token.replace('Ġ', ' ').replace('▁', ' ') for token in tokens]
        
        # Create bar plot
        fig = go.Figure()
        
        # Add bars with custom colors
        colors = px.colors.diverging.RdBu_r
        
        # Normalize scores for color mapping
        score_min, score_max = scores.min(), scores.max()
        if score_max - score_min > 1e-8:
            norm_scores = (scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.zeros_like(scores)
        
        # Map normalized scores to colors
        bar_colors = []
        for norm_score in norm_scores:
            color_idx = int(norm_score * (len(colors) - 1))
            bar_colors.append(colors[color_idx])
        
        fig.add_trace(go.Bar(
            x=list(range(len(tokens))),
            y=scores,
            text=display_tokens,
            textposition='outside',
            marker_color=bar_colors,
            hovertemplate='<b>Token:</b> %{text}<br><b>Score:</b> %{y:.4f}<extra></extra>',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Token Position",
            yaxis_title="Attribution Score",
            width=width,
            height=height,
            font=dict(size=self.font_size),
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(tokens))),
                ticktext=display_tokens,
                tickangle=45
            )
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive visualization saved to {save_path}")
        
        return fig
    
    def create_rich_panel(
        self,
        tokens: List[str],
        scores: Union[torch.Tensor, np.ndarray],
        title: str = "Attribution Visualization",
        threshold: float = 0.1,
        max_width: int = 80
    ) -> Panel:
        """
        Create a rich panel visualization for terminal display.
        
        Args:
            tokens: List of tokens
            scores: Attribution scores
            title: Panel title
            threshold: Minimum score to highlight
            max_width: Maximum width for text wrapping
            
        Returns:
            Rich Panel object
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        
        # Normalize scores
        score_min, score_max = scores.min(), scores.max()
        if score_max - score_min > 1e-8:
            norm_scores = (scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.zeros_like(scores)
        
        # Create text with styling
        text = Text()
        current_line_length = 0
        
        for token, score, norm_score in zip(tokens, scores, norm_scores):
            # Clean token for display
            display_token = token.replace('Ġ', ' ').replace('▁', ' ')
            
            # Check if we need a new line
            if current_line_length + len(display_token) > max_width:
                text.append("\n")
                current_line_length = 0
            
            # Determine style based on score
            if abs(score) >= threshold:
                if score > 0:
                    if norm_score > 0.8:
                        style = "bold red on white"
                    elif norm_score > 0.6:
                        style = "red on white"
                    else:
                        style = "red"
                else:
                    if norm_score < 0.2:
                        style = "bold blue on white"
                    elif norm_score < 0.4:
                        style = "blue on white"
                    else:
                        style = "blue"
            else:
                style = "dim"
            
            # Add token
            text.append(display_token, style=style)
            text.append(" ")
            current_line_length += len(display_token) + 1
        
        # Create panel
        panel = Panel(
            text,
            title=title,
            title_align="left",
            box=box.ROUNDED,
            expand=False
        )
        
        return panel
    
    def create_comparison_plot(
        self,
        token_score_pairs: Dict[str, Tuple[List[str], Union[torch.Tensor, np.ndarray]]],
        title: str = "Attribution Comparison",
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a comparison plot of multiple attribution methods.
        
        Args:
            token_score_pairs: Dict mapping method names to (tokens, scores) tuples
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size override
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = (self.figsize[0], self.figsize[1] * len(token_score_pairs) / 2)
        
        n_methods = len(token_score_pairs)
        fig, axes = plt.subplots(n_methods, 1, figsize=figsize, sharex=True)
        
        if n_methods == 1:
            axes = [axes]
        
        # Get reference tokens (from first method)
        first_method = list(token_score_pairs.keys())[0]
        ref_tokens = token_score_pairs[first_method][0]
        
        for idx, (method_name, (tokens, scores)) in enumerate(token_score_pairs.items()):
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            
            # Clean tokens
            display_tokens = [token.replace('Ġ', ' ').replace('▁', ' ') for token in tokens]
            
            # Create bar plot
            bars = axes[idx].bar(range(len(tokens)), scores, alpha=0.7)
            
            # Color bars based on attribution
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            for bar, norm_score in zip(bars, norm_scores):
                if norm_score > 0.5:
                    color = plt.cm.Reds(norm_score)
                else:
                    color = plt.cm.Blues(1 - norm_score)
                bar.set_color(color)
            
            axes[idx].set_ylabel(method_name, fontsize=self.font_size)
            axes[idx].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[idx].grid(True, alpha=0.3)
            
            # Only show x-labels on bottom plot
            if idx == n_methods - 1:
                axes[idx].set_xticks(range(len(tokens)))
                axes[idx].set_xticklabels(display_tokens, rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=self.font_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_heatmap_matrix(
        self,
        tokens: List[str],
        scores_matrix: Union[torch.Tensor, np.ndarray],
        method_names: List[str],
        title: str = "Attribution Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap matrix showing attributions across methods and tokens.
        
        Args:
            tokens: List of tokens
            scores_matrix: Matrix of scores [n_methods, n_tokens]
            method_names: Names of attribution methods
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if isinstance(scores_matrix, torch.Tensor):
            scores_matrix = scores_matrix.detach().cpu().numpy()
        
        # Clean tokens for display
        display_tokens = [token.replace('Ġ', ' ').replace('▁', ' ') for token in tokens]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            scores_matrix,
            cmap=self.colormap,
            aspect='auto',
            interpolation='nearest'
        )
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=self.font_size-1)
        ax.set_yticks(range(len(method_names)))
        ax.set_yticklabels(method_names, fontsize=self.font_size)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution Score', fontsize=self.font_size)
        
        # Add text annotations
        for i in range(len(method_names)):
            for j in range(len(tokens)):
                text = ax.text(j, i, f'{scores_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title(title, fontsize=self.font_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_table(
        self,
        attribution_results: Dict[str, Tuple[List[str], Union[torch.Tensor, np.ndarray]]],
        top_k: int = 5
    ) -> Table:
        """
        Create a summary table showing top attributed tokens for each method.
        
        Args:
            attribution_results: Dict of method name to (tokens, scores)
            top_k: Number of top tokens to show per method
            
        Returns:
            Rich Table object
        """
        table = Table(title=f"Top {top_k} Attributed Tokens by Method")
        table.add_column("Method", style="bold")
        table.add_column("Positive Attributions", style="red")
        table.add_column("Negative Attributions", style="blue")
        
        for method_name, (tokens, scores) in attribution_results.items():
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            
            # Clean tokens
            display_tokens = [token.replace('Ġ', ' ').replace('▁', ' ') for token in tokens]
            
            # Get top positive and negative
            token_score_pairs = list(zip(display_tokens, scores))
            
            # Sort by score
            positive_pairs = [(t, s) for t, s in token_score_pairs if s > 0]
            negative_pairs = [(t, s) for t, s in token_score_pairs if s < 0]
            
            positive_pairs.sort(key=lambda x: x[1], reverse=True)
            negative_pairs.sort(key=lambda x: x[1])
            
            # Format top tokens
            top_positive = [f"{token} ({score:.3f})" for token, score in positive_pairs[:top_k]]
            top_negative = [f"{token} ({score:.3f})" for token, score in negative_pairs[:top_k]]
            
            table.add_row(
                method_name,
                "\n".join(top_positive) if top_positive else "None",
                "\n".join(top_negative) if top_negative else "None"
            )
        
        return table