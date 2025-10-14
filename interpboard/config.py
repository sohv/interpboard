"""
Configuration settings for the LLM interpretability library.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    model_name: str = "gpt2"
    device: str = "auto"  # "auto", "cuda", "cpu"
    torch_dtype: str = "auto"  # "auto", "float16", "float32"
    max_length: int = 512
    use_cache: bool = True
    low_cpu_mem_usage: bool = True
    
    def get_torch_dtype(self) -> Optional[torch.dtype]:
        """Convert string dtype to torch dtype."""
        if self.torch_dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        elif self.torch_dtype == "float16":
            return torch.float16
        elif self.torch_dtype == "float32":
            return torch.float32
        else:
            return None


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    # Color schemes
    heatmap_colormap: str = "RdYlBu_r"
    attention_colormap: str = "Blues"
    attribution_colormap: str = "RdBu_r"
    
    # Figure settings
    figure_size: tuple = (12, 8)
    dpi: int = 100
    font_size: int = 12
    title_font_size: int = 14
    
    # Interactive settings
    use_plotly: bool = True
    show_colorbar: bool = True
    
    # Text visualization
    max_tokens_display: int = 50
    token_spacing: float = 0.1
    highlight_threshold: float = 0.1


@dataclass
class PatchingConfig:
    """Configuration for activation patching experiments."""
    # Patching strategies
    default_strategy: str = "zero"  # "zero", "mean", "random", "noise"
    noise_std: float = 0.1
    
    # Component targeting
    target_layers: Optional[list] = None
    target_heads: Optional[list] = None
    target_components: list = field(default_factory=lambda: ["attention", "mlp"])
    
    # Metrics
    metrics: list = field(default_factory=lambda: ["logit_diff", "prob_diff", "rank_change"])
    
    # Performance
    batch_size: int = 8
    use_cache: bool = True


@dataclass
class AttributionConfig:
    """Configuration for attribution methods."""
    # Methods to use
    methods: list = field(default_factory=lambda: ["gradient", "integrated_gradient", "attention"])
    
    # Integrated gradients settings
    ig_steps: int = 50
    ig_baseline: str = "zero"  # "zero", "mask", "random"
    
    # Gradient settings
    normalize_gradients: bool = True
    absolute_values: bool = False
    
    # Attention settings
    attention_rollout: bool = True
    attention_head_fusion: str = "mean"  # "mean", "max", "min"


@dataclass
class CircuitConfig:
    """Configuration for circuit analysis."""
    # Logit lens settings
    logit_lens_layers: str = "all"  # "all", "even", "odd", or list of layer indices
    
    # Neuron analysis
    activation_threshold: float = 0.5
    top_k_neurons: int = 100
    
    # Circuit discovery
    intervention_threshold: float = 0.1
    min_circuit_size: int = 2
    max_circuit_size: int = 20


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""
    # Output settings
    save_results: bool = True
    results_dir: str = "./results"
    experiment_name: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    # Performance
    num_workers: int = 1
    use_multiprocessing: bool = False
    seed: int = 42


@dataclass
class Config:
    """Main configuration class combining all settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        viz_config = VisualizationConfig(**config_dict.get("visualization", {}))
        patch_config = PatchingConfig(**config_dict.get("patching", {}))
        attr_config = AttributionConfig(**config_dict.get("attribution", {}))
        circuit_config = CircuitConfig(**config_dict.get("circuit", {}))
        exp_config = ExperimentConfig(**config_dict.get("experiment", {}))
        
        return cls(
            model=model_config,
            visualization=viz_config,
            patching=patch_config,
            attribution=attr_config,
            circuit=circuit_config,
            experiment=exp_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "visualization": self.visualization.__dict__,
            "patching": self.patching.__dict__,
            "attribution": self.attribution.__dict__,
            "circuit": self.circuit.__dict__,
            "experiment": self.experiment.__dict__
        }


# Default configurations for common models
DEFAULT_CONFIGS = {
    "gpt2": Config(
        model=ModelConfig(
            model_name="gpt2",
            max_length=1024
        )
    ),
    "gpt2-medium": Config(
        model=ModelConfig(
            model_name="gpt2-medium",
            max_length=1024
        )
    ),
    "gpt2-large": Config(
        model=ModelConfig(
            model_name="gpt2-large",
            max_length=1024,
            torch_dtype="float16"
        )
    ),
    "llama": Config(
        model=ModelConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            max_length=2048,
            torch_dtype="float16"
        )
    ),
    "mistral": Config(
        model=ModelConfig(
            model_name="mistralai/Mistral-7B-v0.1",
            max_length=2048,
            torch_dtype="float16"
        )
    )
}


def get_config(model_name: str = "gpt2") -> Config:
    """Get default configuration for a model."""
    if model_name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[model_name]
    else:
        # Create default config with specified model name
        config = Config()
        config.model.model_name = model_name
        return config


def load_config_from_file(filepath: str) -> Config:
    """Load configuration from file."""
    import json
    import yaml
    from pathlib import Path
    
    filepath = Path(filepath)
    
    if filepath.suffix == ".json":
        with open(filepath, "r") as f:
            config_dict = json.load(f)
    elif filepath.suffix in [".yaml", ".yml"]:
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {filepath.suffix}")
    
    return Config.from_dict(config_dict)


def save_config_to_file(config: Config, filepath: str) -> None:
    """Save configuration to file."""
    import json
    import yaml
    from pathlib import Path
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    if filepath.suffix == ".json":
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)
    elif filepath.suffix in [".yaml", ".yml"]:
        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported config file format: {filepath.suffix}")


# Environment variable overrides
def apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""
    # Model overrides
    if "INTERPRET_MODEL_NAME" in os.environ:
        config.model.model_name = os.environ["INTERPRET_MODEL_NAME"]
    
    if "INTERPRET_DEVICE" in os.environ:
        config.model.device = os.environ["INTERPRET_DEVICE"]
    
    if "INTERPRET_MAX_LENGTH" in os.environ:
        config.model.max_length = int(os.environ["INTERPRET_MAX_LENGTH"])
    
    # Results directory override
    if "INTERPRET_RESULTS_DIR" in os.environ:
        config.experiment.results_dir = os.environ["INTERPRET_RESULTS_DIR"]
    
    # Logging level override
    if "INTERPRET_LOG_LEVEL" in os.environ:
        config.experiment.log_level = os.environ["INTERPRET_LOG_LEVEL"]
    
    return config


# Global config instance
_global_config = None


def get_global_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = get_config()
        _global_config = apply_env_overrides(_global_config)
    return _global_config


def set_global_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_global_config() -> None:
    """Reset the global configuration to default."""
    global _global_config
    _global_config = None