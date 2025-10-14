# InterpBoard

A comprehensive, modular Python library for interpreting and analyzing transformer language models. Built for researchers and developers who want to understand how LLMs work under the hood.

## 🚀 Quick Start

```python
from interpboard.dashboards import create_unified_dashboard

# Create dashboards for any transformer model
attribution_dashboard, ablation_dashboard = create_unified_dashboard("gpt2")

# Analyze text with attribution methods
text = "The Eiffel Tower is located in Paris, France."
attribution_result = attribution_dashboard.analyze(
    text,
    methods=["integrated_gradients", "attention_rollout"],
    visualize=True,
    save_html=True
)

# Run comprehensive ablation analysis
ablation_result = ablation_dashboard.analyze(
    text,
    analyses=["patching", "logit_lens", "head_ablation"],
    visualize=True
)
```

## ✨ Features

### 🎯 Attribution Methods
- **Gradient-based**: Vanilla gradients, Gradient×Input, Integrated Gradients
- **Attention-based**: Attention rollout, Attention flow
- **Rich visualizations**: Token heatmaps, interactive HTML, comparison plots

### 🔧 Activation Patching & Causal Analysis
- **Systematic ablation**: Test any model component (attention heads, MLP layers, embeddings)
- **Causal tracing**: Trace information flow for factual associations
- **Critical component detection**: Automatically find the most important components

### 🧠 Mechanistic Interpretability
- **Logit lens**: Analyze layer-by-layer predictions and convergence
- **Neuron analysis**: Individual neuron behavior and max-activating inputs
- **Attention head ablation**: Systematic head-by-head analysis
- **Circuit discovery**: Tools for finding computational circuits

### 📊 Visualization
- **Publication-ready plots**: Matplotlib and Plotly integration
- **Interactive visualizations**: Rich terminal output and HTML exports
- **Multi-format support**: Static images, interactive plots, text overlays

### 🏗️ Modular Architecture
- **Model agnostic**: Works with GPT-2, GPT-Neo, LLaMA, Mistral, and more
- **Flexible configuration**: Extensive customization options
- **Easy integration**: Simple APIs for both quick analysis and deep research

## � Installation

```bash
# Clone the repository
git clone https://github.com/your-username/interpboard.git
cd interpboard

Users can install with:
```bash
pip install interpboard
```

And use the CLI:
```bash
interpboard analyze --model gpt2 --text "Your text here"
```
```

## � Examples

### Quick Attribution Analysis
```python
from interpboard.attribution import GradientAttributor, AttributionVisualizer

# Load your model
model, tokenizer = load_model_and_tokenizer("gpt2")

# Analyze token importance
attributor = GradientAttributor(model, tokenizer)
result = attributor.integrated_gradients("The capital of France is Paris.")

# Visualize results
visualizer = AttributionVisualizer()
visualizer.plot_token_heatmap(result, title="Token Attribution")
```

### Causal Tracing
```python
from interpboard.patching import CausalTracer

with CausalTracer(model, tokenizer) as tracer:
    result = tracer.trace_causal_effect(
        text="The Eiffel Tower is located in",
        subject_tokens=["Eiffel", "Tower"],
        target_token_position=-1
    )
    
    print(f"Restoration effect: {result.restoration_effect}")
```

### Attention Analysis
```python
from interpboard.circuits import AttentionHeadAblator

with AttentionHeadAblator(model, tokenizer) as ablator:
    results = ablator.systematic_head_ablation(
        "The cat sat on the mat.",
        layers=[8, 9, 10, 11],
        heads=list(range(12))
    )
    
    critical_heads = ablator.find_critical_heads(results)
    print(f"Critical heads: {critical_heads}")
```

## 📚 Documentation

- **[API Reference](docs/api.md)**: Detailed API documentation
- **[Examples](examples/)**: Comprehensive example notebooks
- **[Configuration Guide](docs/configuration.md)**: Advanced configuration options
- **[Model Support](docs/models.md)**: Supported models and architectures

## 🛠️ Supported Models

The library works with any transformer model from Hugging Face:

- **GPT family**: GPT-2, GPT-Neo, GPT-J
- **LLaMA family**: LLaMA, LLaMA-2, Code Llama
- **Mistral family**: Mistral 7B, Mixtral
- **Other architectures**: BERT, RoBERTa, T5 (experimental)

## 🏆 Key Applications

- **🔬 Research**: Mechanistic interpretability studies, circuit discovery
- **🛡️ Safety**: Understanding model behavior, bias detection
- **🎯 Debugging**: Model troubleshooting, performance analysis
- **📚 Education**: Teaching how LLMs work, visualization for papers

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on top of:
- [PyTorch](https://pytorch.org/) for deep learning
- [Transformers](https://huggingface.co/transformers/) for model loading
- [Matplotlib](https://matplotlib.org/) and [Plotly](https://plotly.com/) for visualization
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output

Inspired by research in mechanistic interpretability from:
- Anthropic's work on [Transformer Circuits](https://transformer-circuits.pub/)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) by Neel Nanda
- [Baukit](https://github.com/davidbau/baukit) intervention tools

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/interpboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/interpboard/discussions)
- **Email**: your-email@example.com

---

**Happy interpreting! 🧠✨**

## 📁 Library Structure

```
interpret_llm/
│
├── patching/                 # Activation patching and intervention tools
│   ├── activ_patch.py       # Core activation patching functionality
│   ├── causal_tracing.py    # Causal tracing and path patching
│   └── utils.py             # Patching utilities
│
├── attribution/              # Attribution and gradient methods
│   ├── gradients.py         # Gradient-based attribution methods
│   ├── attention_rollout.py # Attention-based attribution
│   └── viz.py               # Attribution visualization
│
├── visualization/            # Visualization components
│   ├── text_overlay.py      # Text heatmap overlays
│   ├── heatmap.py           # General heatmap utilities
│   └── styles/              # CSS and styling for visualizations
│
├── circuits/                 # Mechanistic interpretability tools
│   ├── logit_lens.py        # Logit lens implementation
│   ├── neuron_analysis.py   # Neuron activation analysis
│   └── head_ablation.py     # Attention head ablation
│
├── examples/                 # Example notebooks and scripts
├── config.py                # Configuration management
├── utils.py                 # Core utilities
└── dashboards.py            # Main dashboard interfaces
```

## 🔬 Advanced Examples

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

## 🎨 Visualization Gallery

The library provides rich, interactive visualizations:

- **Token Heatmaps**: Color-coded importance scores
- **Activation Flow**: Layer-by-layer processing visualization  
- **Attention Patterns**: Head-specific attention visualizations
- **Circuit Diagrams**: Mechanistic circuit representations
- **Comparative Plots**: Before/after intervention comparisons

## 🤝 Supported Models

- **GPT-2** (all sizes)
- **GPT-Neo/GPT-J**
- **LLaMA** (with appropriate access)
- **Mistral**
- **Any HuggingFace Transformer** with attention mechanisms

## 🧪 Research Applications

This toolkit is designed for:

- **Mechanistic Interpretability Research**
- **AI Safety and Alignment Studies**  
- **Model Debugging and Analysis**
- **Educational Exploration of LLMs**
- **Comparative Model Analysis**

## 📖 Documentation

Full documentation is available at [link-to-docs].

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Tutorials](docs/tutorials/)
- [Examples](examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Inspired by and builds upon:
- **TransformerLens** - Mechanistic interpretability framework
- **Captum** - Model interpretability for PyTorch  
- **OpenAI's Circuits** - Pioneering mechanistic interpretability research
- **Anthropic's Research** - Causal tracing and compositional explanations

## 📊 Citation

If you use this library in your research, please cite:

```bibtex
@software{interpret_llm,
  title={LLM Interpretability Dashboard: A Comprehensive Toolkit for Understanding Large Language Models},
  author={LLM Interpretability Team},
  year={2024},
  url={https://github.com/yourusername/interpret-llm}
}
```