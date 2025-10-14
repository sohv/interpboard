"""
Visualization utilities for attribution results.

This module provides functions to visualize attribution scores in various formats
including text overlays, heatmaps, and interactive plots.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns
from rich.console import Console
from rich.text import Text
from rich.table import Table
import logging

from .gradients import AttributionResult

logger = logging.getLogger(__name__)


class AttributionVisualizer:
    """Visualizes attribution results in various formats."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        colormap: str = "RdBu_r",
        font_size: int = 12
    ):
        self.figsize = figsize
        self.colormap = colormap
        self.font_size = font_size
        self.console = Console()
    
    def plot_token_heatmap(
        self,
        attribution_result: AttributionResult,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_values: bool = True,
        normalize: bool = True
    ) -> plt.Figure:
        """
        Create a heatmap visualization of token attributions.
        
        Args:
            attribution_result: Attribution result to visualize
            title: Plot title
            save_path: Path to save the plot
            show_values: Whether to show attribution values on tokens
            normalize: Whether to normalize attributions for visualization
            
        Returns:
            Matplotlib figure
        """
        attributions = attribution_result.attributions.detach().cpu().numpy()
        tokens = attribution_result.tokens
        
        if normalize:
            attr_min, attr_max = attributions.min(), attributions.max()
            if attr_max - attr_min > 1e-8:
                attributions = (attributions - attr_min) / (attr_max - attr_min)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap data
        heatmap_data = attributions.reshape(1, -1)
        
        # Plot heatmap
        im = ax.imshow(
            heatmap_data,
            cmap=self.colormap,
            aspect='auto',
            vmin=attributions.min(),
            vmax=attributions.max()
        )
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=self.font_size)
        ax.set_yticks([])
        
        # Add values on tokens if requested
        if show_values:
            for i, value in enumerate(attributions):
                text_color = 'white' if abs(value - attributions.mean()) > 0.5 * attributions.std() else 'black'
                ax.text(i, 0, f'{value:.3f}', ha='center', va='center', 
                       color=text_color, fontsize=self.font_size-2)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution Score', fontsize=self.font_size)
        
        # Set title
        if title is None:
            title = f"Token Attribution ({attribution_result.method_name})"
        ax.set_title(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_text_overlay(
        self,
        attribution_result: AttributionResult,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        background_alpha: float = 0.7
    ) -> plt.Figure:
        """
        Create a text overlay visualization with colored backgrounds.
        
        Args:
            attribution_result: Attribution result to visualize
            title: Plot title
            save_path: Path to save the plot
            background_alpha: Alpha value for background colors
            
        Returns:
            Matplotlib figure
        """
        attributions = attribution_result.attributions.detach().cpu().numpy()
        tokens = attribution_result.tokens
        
        # Normalize attributions for color mapping
        attr_min, attr_max = attributions.min(), attributions.max()
        if attr_max - attr_min > 1e-8:
            norm_attributions = (attributions - attr_min) / (attr_max - attr_min)
        else:
            norm_attributions = np.zeros_like(attributions)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Choose colormap
        cmap = plt.cm.get_cmap(self.colormap)
        
        # Plot each token with colored background
        x_pos = 0
        max_line_width = 80  # Characters per line
        current_line_width = 0
        y_pos = 0
        
        for i, (token, attribution, norm_attr) in enumerate(zip(tokens, attributions, norm_attributions)):
            # Clean token for display
            display_token = token.replace('Ġ', ' ').replace('▁', ' ')
            
            # Check if we need a new line
            if current_line_width + len(display_token) > max_line_width:
                y_pos -= 1
                x_pos = 0
                current_line_width = 0
            
            # Get color for this token
            color = cmap(norm_attr)
            
            # Add background rectangle
            rect = Rectangle(
                (x_pos, y_pos - 0.4), len(display_token) * 0.1, 0.8,
                facecolor=color, alpha=background_alpha, edgecolor='none'
            )
            ax.add_patch(rect)
            
            # Add text
            text_color = 'white' if norm_attr < 0.5 else 'black'
            ax.text(
                x_pos + len(display_token) * 0.05, y_pos,
                display_token,
                fontsize=self.font_size,
                color=text_color,
                ha='center', va='center'
            )
            
            x_pos += len(display_token) * 0.1 + 0.05  # Small spacing
            current_line_width += len(display_token) + 1
        
        # Set axis properties
        ax.set_xlim(-0.5, max_line_width * 0.1)
        ax.set_ylim(y_pos - 1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=attr_min, vmax=attr_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Attribution Score', fontsize=self.font_size)
        
        # Set title
        if title is None:
            title = f"Text Attribution Overlay ({attribution_result.method_name})"
        fig.suptitle(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_attributions(
        self,
        attribution_results: Dict[str, AttributionResult],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare multiple attribution methods side by side.
        
        Args:
            attribution_results: Dictionary of attribution results
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_methods = len(attribution_results)
        if n_methods == 0:
            raise ValueError("No attribution results provided")
        
        # Get common tokens (use first result as reference)
        first_result = list(attribution_results.values())[0]
        tokens = first_result.tokens
        n_tokens = len(tokens)
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_methods, 1, figsize=(self.figsize[0], self.figsize[1] * n_methods / 2))
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, result) in enumerate(attribution_results.items()):
            attributions = result.attributions.detach().cpu().numpy()
            
            # Normalize for consistent color scale
            attr_min, attr_max = attributions.min(), attributions.max()
            if attr_max - attr_min > 1e-8:
                norm_attributions = (attributions - attr_min) / (attr_max - attr_min)
            else:
                norm_attributions = np.zeros_like(attributions)
            
            # Create heatmap
            heatmap_data = norm_attributions.reshape(1, -1)
            im = axes[idx].imshow(
                heatmap_data,
                cmap=self.colormap,
                aspect='auto',
                vmin=0, vmax=1
            )
            
            # Set labels
            axes[idx].set_xticks(range(len(tokens)))
            if idx == n_methods - 1:  # Only show x-labels on bottom plot
                axes[idx].set_xticklabels(tokens, rotation=45, ha='right', fontsize=self.font_size-1)
            else:
                axes[idx].set_xticklabels([])
            
            axes[idx].set_yticks([])
            axes[idx].set_ylabel(method_name, fontsize=self.font_size)
            
            # Add colorbar for each method
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Normalized Attribution', fontsize=self.font_size-1)
        
        # Set main title
        if title is None:
            title = "Attribution Method Comparison"
        fig.suptitle(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def rich_text_display(
        self,
        attribution_result: AttributionResult,
        threshold: float = 0.1,
        show_scores: bool = True
    ) -> None:
        """
        Display attribution results in the terminal using rich formatting.
        
        Args:
            attribution_result: Attribution result to display
            threshold: Minimum attribution to highlight
            show_scores: Whether to show numerical scores
        """
        attributions = attribution_result.attributions.detach().cpu().numpy()
        tokens = attribution_result.tokens
        
        # Normalize attributions
        attr_min, attr_max = attributions.min(), attributions.max()
        if attr_max - attr_min > 1e-8:
            norm_attributions = (attributions - attr_min) / (attr_max - attr_min)
        else:
            norm_attributions = np.zeros_like(attributions)
        
        # Create rich text
        text = Text()
        
        for token, attribution, norm_attr in zip(tokens, attributions, norm_attributions):
            # Clean token for display
            display_token = token.replace('Ġ', ' ').replace('▁', ' ')
            
            # Determine style based on attribution
            if abs(attribution) >= threshold:
                if attribution > 0:
                    style = "bold red" if norm_attr > 0.7 else "red"
                else:
                    style = "bold blue" if norm_attr < 0.3 else "blue"
            else:
                style = "dim"
            
            # Add token with optional score
            if show_scores:
                token_text = f"{display_token}({attribution:.3f})"
            else:
                token_text = display_token
            
            text.append(token_text + " ", style=style)
        
        # Print header and text
        self.console.print(f"\n[bold]{attribution_result.method_name} Attribution:[/bold]")
        self.console.print(text)
        
        # Print summary
        self.console.print(f"\nSummary:")
        self.console.print(f"  Max attribution: {attributions.max():.4f}")
        self.console.print(f"  Min attribution: {attributions.min():.4f}")
        self.console.print(f"  Mean attribution: {attributions.mean():.4f}")
        self.console.print(f"  Std attribution: {attributions.std():.4f}")
    
    def create_attribution_table(
        self,
        attribution_result: AttributionResult,
        top_k: int = 10,
        show_positive: bool = True,
        show_negative: bool = True
    ) -> Table:
        """
        Create a rich table showing top attributed tokens.
        
        Args:
            attribution_result: Attribution result to display
            top_k: Number of top tokens to show
            show_positive: Whether to show most positive attributions
            show_negative: Whether to show most negative attributions
            
        Returns:
            Rich Table object
        """
        attributions = attribution_result.attributions.detach().cpu().numpy()
        tokens = attribution_result.tokens
        
        # Create table
        table = Table(title=f"{attribution_result.method_name} - Top {top_k} Attributions")
        table.add_column("Rank", style="dim")
        table.add_column("Token", style="bold")
        table.add_column("Attribution", justify="right")
        table.add_column("Type", style="dim")
        
        # Get top positive and negative attributions
        token_attr_pairs = list(zip(tokens, attributions))
        
        rows = []
        
        if show_positive:
            # Top positive
            sorted_positive = sorted(token_attr_pairs, key=lambda x: x[1], reverse=True)
            for i, (token, attr) in enumerate(sorted_positive[:top_k]):
                if attr > 0:
                    rows.append((i+1, token.replace('Ġ', ' ').replace('▁', ' '), f"{attr:.4f}", "Positive"))
        
        if show_negative:
            # Top negative
            sorted_negative = sorted(token_attr_pairs, key=lambda x: x[1])
            for i, (token, attr) in enumerate(sorted_negative[:top_k]):
                if attr < 0:
                    rows.append((i+1, token.replace('Ġ', ' ').replace('▁', ' '), f"{attr:.4f}", "Negative"))
        
        # Sort combined results by absolute value
        rows.sort(key=lambda x: abs(float(x[2])), reverse=True)
        
        # Add rows to table
        for rank, token, attr, attr_type in rows[:top_k]:
            color = "red" if attr_type == "Positive" else "blue"
            table.add_row(
                str(rank),
                token,
                f"[{color}]{attr}[/{color}]",
                attr_type
            )
        
        return table
    
    def plot_attribution_distribution(
        self,
        attribution_results: Dict[str, AttributionResult],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of attribution scores across methods.
        
        Args:
            attribution_results: Dictionary of attribution results
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Collect all attributions
        all_attributions = {}
        for method_name, result in attribution_results.items():
            attributions = result.attributions.detach().cpu().numpy()
            all_attributions[method_name] = attributions
        
        # Plot histograms
        for method_name, attributions in all_attributions.items():
            axes[0].hist(attributions, alpha=0.7, label=method_name, bins=30)
        
        axes[0].set_xlabel('Attribution Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Attribution Distributions')
        axes[0].legend()
        
        # Plot box plots
        data_for_box = list(all_attributions.values())
        labels = list(all_attributions.keys())
        
        bp = axes[1].boxplot(data_for_box, labels=labels, patch_artist=True)
        
        # Color box plots
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_box)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1].set_xlabel('Method')
        axes[1].set_ylabel('Attribution Score')
        axes[1].set_title('Attribution Score Distributions')
        axes[1].tick_params(axis='x', rotation=45)
        
        if title:
            fig.suptitle(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig