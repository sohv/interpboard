# InterpBoard

A comprehensive, modular Python library for interpreting and analyzing transformer language models. Built for researchers and developers who want to understand how LLMs work under the hood.

## Quick Start

```python
from interpboard.dashboards import create_unified_dashboard

# create dashboards for any transformer model
attribution_dashboard, ablation_dashboard = create_unified_dashboard("gpt2")

# analyze text with attribution methods
text = "The Eiffel Tower is located in Paris, France."
attribution_result = attribution_dashboard.analyze(
    text,
    methods=["integrated_gradients", "attention_rollout"],
    visualize=True,
    interactive=True
)

# run comprehensive ablation analysis
ablation_result = ablation_dashboard.analyze(
    text,
    analyses=["patching", "logit_lens", "head_ablation"],
    visualize=True
)
```

## Features

### Attribution Methods
- **Gradient-based**: Vanilla gradients, Gradient×Input, Integrated Gradients
- **Attention-based**: Attention rollout, Attention flow
- **Rich visualizations**: Token heatmaps, interactive HTML, comparison plots

### Activation Patching & Causal Analysis
- **Systematic ablation**: Test any model component (attention heads, MLP layers, embeddings)
- **Causal tracing**: Trace information flow for factual associations
- **Critical component detection**: Automatically find the most important components

### Mechanistic Interpretability
- **Logit lens**: Analyze layer-by-layer predictions and convergence
- **Neuron analysis**: Individual neuron behavior and max-activating inputs
- **Attention head ablation**: Systematic head-by-head analysis
- **Circuit discovery**: Tools for finding computational circuits

### Visualization
- **Publication-ready plots**: Matplotlib and Plotly integration
- **Interactive visualizations**: Rich terminal output and HTML exports
- **Multi-format support**: Static images, interactive plots, text overlays

### Modular Architecture
- **Model agnostic**: Works with GPT-2, GPT-Neo, LLaMA, Mistral, and more
- **Flexible configuration**: Extensive customization options
- **Easy integration**: Simple APIs for both quick analysis and deep research

## Installation

```bash
# Clone the repository
git clone https://github.com/sohv/interpboard.git
cd interpboard

Users can install with:
```bash
pip install interpboard
```

And use the CLI:
```bash
interpboard analyze --model gpt2 --text "Your text here"
```

## Examples

### Quick Attribution Analysis
```python
from interpboard.dashboards import create_unified_dashboard

# Create unified dashboard (automatically loads model)
attribution_dashboard, ablation_dashboard = create_unified_dashboard("gpt2")

# Analyze text with multiple methods and interactive visualizations
result = attribution_dashboard.analyze(
    "The capital of France is Paris.",
    methods=["integrated_gradients", "attention_rollout"],
    visualize=True,
    interactive=True  # Creates interactive Plotly visualizations
)

# Compare multiple texts
comparison_results = attribution_dashboard.compare_texts([
    "The capital of France is Paris.",
    "London is the capital of England."
], method="integrated_gradients", interactive=True)
```

### Causal Tracing & Ablation Analysis
```python
# Use the ablation dashboard for comprehensive patching analysis
patch_result = ablation_dashboard.patch_activations(
    "The Eiffel Tower is located in Paris.",
    layer_range=(4, 8),  # Focus on middle layers
    visualize=True
)

# Run causal tracing experiments
trace_result = ablation_dashboard.causal_trace(
    "The Eiffel Tower is located in",
    subject_tokens=["Eiffel", "Tower"],
    target_position=-1,
    visualize=True
)

print(f"Critical components found: {len(patch_result.patch_effects)}")
```

### Attention Analysis
```python
# Attention analysis is integrated into the attribution dashboard
result = attribution_dashboard.analyze(
    "The cat sat on the mat.",
    methods=["attention_rollout"],  # Focus on attention-based methods
    visualize=True,
    interactive=True
)

# For detailed attention head analysis, use the ablation dashboard
head_results = ablation_dashboard.analyze_attention_heads(
    "The cat sat on the mat.",
    layers=[8, 9, 10, 11],
    visualize=True
)

print(f"Attention patterns analyzed across {len(head_results.head_effects)} heads")
```

## Documentation

- **[API Reference](docs/api.md)**: Detailed API documentation
- **[Examples](examples/)**: Comprehensive example notebooks
- **[Configuration Guide](docs/configuration.md)**: Advanced configuration options
- **[Model Support](docs/models.md)**: Supported models and architectures

## Supported Models

### Model Support Matrix

| Model Family | Example Models | Attribution | Visualization | Logit Lens | Activation Patching | Neuron Analysis | Head Ablation | Status |
|--------------|----------------|-------------|---------------|------------|-------------------|-----------------|---------------|---------|
| **GPT-2** | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` | YES | YES | YES | YES | YES | YES | **Full Support** |
| **LLaMA** | `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-2-70b-hf` | YES | YES | YES | YES | YES | YES | **Full Support** |
| **Code Llama** | `codellama/CodeLlama-7b-hf`, `codellama/CodeLlama-13b-hf` | YES | YES | YES | YES | YES | YES | **Full Support** |
| **Mistral** | `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1` | YES | YES | YES | YES | YES | YES | **Full Support** |
| **GPT-Neo/J** | `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-j-6B` | YES | YES | PARTIAL | NO | NO | PARTIAL | **Limited Support** |
| **OPT** | `facebook/opt-1.3b`, `facebook/opt-2.7b`, `facebook/opt-6.7b` | YES | YES | PARTIAL | NO | NO | PARTIAL | **Limited Support** |
| **BLOOM** | `bigscience/bloom-1b1`, `bigscience/bloom-3b`, `bigscience/bloom-7b1` | PARTIAL | PARTIAL | NO | NO | NO | NO | **Experimental** |
| **T5** | `t5-small`, `t5-base`, `t5-large` | PARTIAL | PARTIAL | NO | NO | NO | NO | **Experimental** |
| **BERT/RoBERTa** | `bert-base-uncased`, `roberta-base` | PARTIAL | PARTIAL | NO | NO | NO | NO | **Experimental** |

### Legend
- **Full Support**: All features work as expected
- **Partial Support**: Basic functionality works, some advanced features may fail
- **Not Supported**: Feature will raise `NotImplementedError`

### Support Details

#### **Fully Supported Models** 
All interpretability methods work seamlessly:
- **GPT-2 Family**: Complete implementation with extensive testing
- **LLaMA Family**: Full support including LLaMA-2 and Code Llama variants
- **Mistral Family**: Complete support for all Mistral 7B variants

#### **Limited Support Models**
Basic attribution and visualization work, but advanced features are limited:
- **GPT-Neo/GPT-J**: Attribution methods work, but activation patching and neuron analysis may fail
- **OPT Models**: Similar limitations to GPT-Neo/J family

#### **Experimental Support**
Only basic functionality available, most advanced features will fail:
- **Encoder-Only Models** (BERT, RoBERTa): Limited to attribution analysis
- **Encoder-Decoder Models** (T5, BART): Partial support with potential issues
- **Other Architectures**: May work for basic attribution but not guaranteed

### Adding New Model Support

To extend support for additional models:
1. The model must be compatible with HuggingFace Transformers
2. Model architecture should follow standard transformer patterns
3. For full support, contribute model-specific component mappings

See our [Model Support Guide](docs/supported_models.md) for detailed information about extending model compatibility.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built on top of:
- [PyTorch](https://pytorch.org/) for deep learning
- [Transformers](https://huggingface.co/transformers/) for model loading
- [Matplotlib](https://matplotlib.org/) and [Plotly](https://plotly.com/) for visualization
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output

Inspired by research in mechanistic interpretability from:
- Anthropic's work on [Transformer Circuits](https://transformer-circuits.pub/)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) by Neel Nanda
- [Baukit](https://github.com/davidbau/baukit) intervention tools

## Advanced Examples

### Circuit Discovery
```python
from interpboard.circuits import CircuitAnalyzer

analyzer = CircuitAnalyzer(model, tokenizer)
circuits = analyzer.discover_circuits(
    dataset=["The capital of France is", "The president of USA is"],
    target_behavior="country_capital"
)
```

### Comparative Ablation
```python
# Compare multiple ablation strategies
ablation.compare_ablations(
    prompt="The Eiffel Tower is located in",
    strategies=["zero", "mean", "random"],
    components=[(8, 4), (9, 2), (10, 1)]  # (layer, head) pairs
)
```