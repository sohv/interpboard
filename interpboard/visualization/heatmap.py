"""
Heatmap visualization utilities for interpretability results.

This module provides comprehensive heatmap visualizations for attention patterns,
activation patterns, and comparative analysis across models and methods.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class HeatmapVisualizer:
    """Creates various heatmap visualizations for interpretability analysis."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        colormap: str = "RdBu_r",
        font_size: int = 10
    ):
        self.figsize = figsize
        self.colormap = colormap
        self.font_size = font_size
    
    def attention_heatmap(
        self,
        attention_weights: Union[torch.Tensor, np.ndarray],
        tokens: List[str],
        title: str = "Attention Heatmap",
        save_path: Optional[str] = None,
        show_values: bool = False,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None
    ) -> plt.Figure:
        """
        Create attention heatmap visualization.
        
        Args:
            attention_weights: Attention weights [seq_len, seq_len] or [heads, seq_len, seq_len]
            tokens: List of tokens
            title: Plot title
            save_path: Path to save figure
            show_values: Whether to show numerical values
            layer_idx: Layer index for title
            head_idx: Head index for title
            
        Returns:
            Matplotlib figure
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Handle multi-head attention
        if attention_weights.ndim == 3:
            if head_idx is not None:
                attention_weights = attention_weights[head_idx]
            else:
                # Average across heads
                attention_weights = attention_weights.mean(axis=0)
        
        # Clean tokens for display
        display_tokens = [token.replace('Ġ', ' ').replace('▁', ' ') for token in tokens]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            attention_weights,
            cmap=self.colormap,
            aspect='equal',
            interpolation='nearest'
        )
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=self.font_size-1)
        ax.set_yticklabels(display_tokens, fontsize=self.font_size-1)
        
        # Add labels
        ax.set_xlabel('Key Tokens', fontsize=self.font_size)
        ax.set_ylabel('Query Tokens', fontsize=self.font_size)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=self.font_size)
        
        # Add values if requested
        if show_values:
            for i in range(len(tokens)):
                for j in range(len(tokens)):
                    text = ax.text(j, i, f'{attention_weights[i, j]:.3f}',
                                 ha="center", va="center", 
                                 color="white" if attention_weights[i, j] > 0.5 else "black",
                                 fontsize=8)
        
        # Set title
        full_title = title
        if layer_idx is not None:
            full_title += f" (Layer {layer_idx}"
            if head_idx is not None:
                full_title += f", Head {head_idx}"
            full_title += ")"
        
        ax.set_title(full_title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def multi_head_attention_grid(
        self,
        attention_weights: Union[torch.Tensor, np.ndarray],
        tokens: List[str],
        title: str = "Multi-Head Attention",
        save_path: Optional[str] = None,
        max_heads: int = 12,
        layer_idx: Optional[int] = None
    ) -> plt.Figure:
        """
        Create a grid of attention heatmaps for multiple heads.
        
        Args:
            attention_weights: Attention weights [heads, seq_len, seq_len]
            tokens: List of tokens
            title: Plot title
            save_path: Path to save figure
            max_heads: Maximum number of heads to display
            layer_idx: Layer index for title
            
        Returns:
            Matplotlib figure
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        if attention_weights.ndim != 3:
            raise ValueError("Expected 3D attention weights [heads, seq_len, seq_len]")
        
        num_heads = min(attention_weights.shape[0], max_heads)
        
        # Calculate grid dimensions
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Clean tokens for display
        display_tokens = [token.replace('Ġ', ' ').replace('▁', ' ') for token in tokens]
        
        for head_idx in range(num_heads):
            row = head_idx // cols
            col = head_idx % cols
            ax = axes[row, col]
            
            # Create heatmap for this head
            im = ax.imshow(
                attention_weights[head_idx],
                cmap=self.colormap,
                aspect='equal',
                interpolation='nearest'
            )
            
            # Set labels only for edge subplots
            if row == rows - 1:  # Bottom row
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticks([])
            
            if col == 0:  # Left column
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(display_tokens, fontsize=8)
            else:
                ax.set_yticks([])
            
            ax.set_title(f'Head {head_idx}', fontsize=self.font_size)
        
        # Hide unused subplots
        for head_idx in range(num_heads, rows * cols):
            row = head_idx // cols
            col = head_idx % cols
            axes[row, col].set_visible(False)
        
        # Add overall title
        full_title = title
        if layer_idx is not None:
            full_title += f" (Layer {layer_idx})"
        fig.suptitle(full_title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def activation_heatmap(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        tokens: Optional[List[str]] = None,
        component_names: Optional[List[str]] = None,
        title: str = "Activation Heatmap",
        save_path: Optional[str] = None,
        transpose: bool = False
    ) -> plt.Figure:
        """
        Create heatmap for activation patterns.
        
        Args:
            activations: Activation values [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
            tokens: List of token names for y-axis
            component_names: List of component names for x-axis
            title: Plot title
            save_path: Path to save figure
            transpose: Whether to transpose the heatmap
            
        Returns:
            Matplotlib figure
        """
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        # Handle batch dimension
        if activations.ndim == 3:
            activations = activations[0]  # Take first batch
        
        if transpose:
            activations = activations.T
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            activations,
            cmap=self.colormap,
            aspect='auto',
            interpolation='nearest'
        )
        
        # Set labels
        if activations.shape[0] <= 50:  # Only show labels if not too many
            if tokens is not None and not transpose:
                display_tokens = [token.replace('Ġ', ' ').replace('▁', ' ') for token in tokens]
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(display_tokens, fontsize=self.font_size-1)
            elif component_names is not None and transpose:
                ax.set_yticks(range(len(component_names)))
                ax.set_yticklabels(component_names, fontsize=self.font_size-1)
        
        if activations.shape[1] <= 50:  # Only show labels if not too many
            if component_names is not None and not transpose:
                ax.set_xticks(range(len(component_names)))
                ax.set_xticklabels(component_names, rotation=45, ha='right', fontsize=self.font_size-1)
            elif tokens is not None and transpose:
                display_tokens = [token.replace('Ġ', ' ').replace('▁', ' ') for token in tokens]
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=self.font_size-1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Value', fontsize=self.font_size)
        
        ax.set_title(title, fontsize=self.font_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def comparison_heatmap(
        self,
        data_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: str = "Comparison Heatmap",
        save_path: Optional[str] = None,
        normalize_rows: bool = False
    ) -> plt.Figure:
        """
        Create comparison heatmap across multiple conditions/methods.
        
        Args:
            data_dict: Dictionary mapping condition names to data arrays
            row_labels: Labels for rows
            col_labels: Labels for columns
            title: Plot title
            save_path: Path to save figure
            normalize_rows: Whether to normalize each row
            
        Returns:
            Matplotlib figure
        """
        # Stack data arrays
        data_list = []
        method_names = []
        
        for method_name, data in data_dict.items():
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            
            # Flatten if multi-dimensional
            if data.ndim > 1:
                data = data.flatten()
            
            data_list.append(data)
            method_names.append(method_name)
        
        # Create matrix
        comparison_matrix = np.array(data_list)
        
        if normalize_rows:
            # Normalize each row
            for i in range(comparison_matrix.shape[0]):
                row = comparison_matrix[i]
                row_min, row_max = row.min(), row.max()
                if row_max - row_min > 1e-8:
                    comparison_matrix[i] = (row - row_min) / (row_max - row_min)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            comparison_matrix,
            cmap=self.colormap,
            aspect='auto',
            interpolation='nearest'
        )
        
        # Set labels
        ax.set_yticks(range(len(method_names)))
        ax.set_yticklabels(method_names, fontsize=self.font_size)
        
        if col_labels is not None:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=self.font_size-1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        label = 'Normalized Value' if normalize_rows else 'Value'
        cbar.set_label(label, fontsize=self.font_size)
        
        ax.set_title(title, fontsize=self.font_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def interactive_heatmap(
        self,
        data: Union[torch.Tensor, np.ndarray],
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        title: str = "Interactive Heatmap",
        save_path: Optional[str] = None,
        width: int = 800,
        height: int = 600
    ) -> go.Figure:
        """
        Create interactive heatmap using Plotly.
        
        Args:
            data: 2D data array
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            title: Plot title
            save_path: Path to save HTML file
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale='RdBu_r',
            hoverongaps=False,
            hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Value:</b> %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            font=dict(size=self.font_size),
            xaxis_title="Columns",
            yaxis_title="Rows"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive heatmap saved to {save_path}")
        
        return fig
    
    def correlation_heatmap(
        self,
        correlation_matrix: Union[torch.Tensor, np.ndarray],
        labels: List[str],
        title: str = "Correlation Heatmap",
        save_path: Optional[str] = None,
        annotate: bool = True
    ) -> plt.Figure:
        """
        Create correlation heatmap with annotations.
        
        Args:
            correlation_matrix: Square correlation matrix
            labels: Labels for both axes
            title: Plot title
            save_path: Path to save figure
            annotate: Whether to show correlation values
            
        Returns:
            Matplotlib figure
        """
        if isinstance(correlation_matrix, torch.Tensor):
            correlation_matrix = correlation_matrix.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Use seaborn for better correlation heatmap
        sns.heatmap(
            correlation_matrix,
            annot=annotate,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={'label': 'Correlation'},
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_title(title, fontsize=self.font_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def layerwise_heatmap(
        self,
        layer_data: List[Union[torch.Tensor, np.ndarray]],
        layer_names: Optional[List[str]] = None,
        tokens: Optional[List[str]] = None,
        title: str = "Layer-wise Activation",
        save_path: Optional[str] = None,
        aggregation: str = "mean"
    ) -> plt.Figure:
        """
        Create heatmap showing patterns across layers.
        
        Args:
            layer_data: List of data arrays for each layer
            layer_names: Names for each layer
            tokens: Token names
            title: Plot title
            save_path: Path to save figure
            aggregation: How to aggregate data ('mean', 'max', 'sum')
            
        Returns:
            Matplotlib figure
        """
        # Process each layer's data
        processed_data = []
        
        for layer_idx, data in enumerate(layer_data):
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            
            # Aggregate over features/dimensions if needed
            if data.ndim > 1:
                if aggregation == "mean":
                    aggregated = data.mean(axis=-1)
                elif aggregation == "max":
                    aggregated = data.max(axis=-1)
                elif aggregation == "sum":
                    aggregated = data.sum(axis=-1)
                else:
                    aggregated = data.mean(axis=-1)
            else:
                aggregated = data
            
            processed_data.append(aggregated)
        
        # Stack into matrix [layers, tokens/positions]
        heatmap_data = np.array(processed_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            heatmap_data,
            cmap=self.colormap,
            aspect='auto',
            interpolation='nearest'
        )
        
        # Set labels
        if layer_names is not None:
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels(layer_names, fontsize=self.font_size)
        else:
            ax.set_ylabel('Layer', fontsize=self.font_size)
        
        if tokens is not None:
            display_tokens = [token.replace('Ġ', ' ').replace('▁', ' ') for token in tokens]
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=self.font_size-1)
        else:
            ax.set_xlabel('Position', fontsize=self.font_size)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{aggregation.title()} Activation', fontsize=self.font_size)
        
        ax.set_title(f'{title} ({aggregation})', fontsize=self.font_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig