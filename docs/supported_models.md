# Supported Models

The LLM Interpretability Dashboard supports a wide range of transformer-based language models through Hugging Face's Transformers library.

## Fully Supported Models

### GPT Family
- **GPT-2** (all sizes: 117M, 345M, 762M, 1.5B)
  - `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
  - Full support for all features
  
- **GPT-Neo** (EleutherAI)
  - `EleutherAI/gpt-neo-125M`, `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-neo-2.7B`
  - Full feature support
  
- **GPT-J** (EleutherAI)
  - `EleutherAI/gpt-j-6B`
  - Full feature support

### LLaMA Family
- **LLaMA** (Meta)
  - `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-2-70b-hf`
  - `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`
  - Requires appropriate access permissions
  
- **Code Llama**
  - `codellama/CodeLlama-7b-hf`, `codellama/CodeLlama-13b-hf`
  - `codellama/CodeLlama-7b-Python-hf`

### Mistral Family
- **Mistral 7B**
  - `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`
  - `mistralai/Mistral-7B-v0.3`

- **Mixtral**
  - `mistralai/Mixtral-8x7B-v0.1`
  - Note: Requires significant memory for full analysis

### Other Decoder Models
- **OPT** (Meta)
  - `facebook/opt-125m`, `facebook/opt-350m`, `facebook/opt-1.3b`, `facebook/opt-2.7b`, `facebook/opt-6.7b`
  
- **BLOOM** (BigScience)
  - `bigscience/bloom-560m`, `bigscience/bloom-1b1`, `bigscience/bloom-3b`, `bigscience/bloom-7b1`

## Experimental Support

### Encoder-Decoder Models
- **T5** (Google)
  - `t5-small`, `t5-base`, `t5-large`
  - Limited support (encoder analysis only)
  
- **BART** (Facebook)
  - `facebook/bart-base`, `facebook/bart-large`
  - Limited support

### Encoder-Only Models
- **BERT** (Google)
  - `bert-base-uncased`, `bert-large-uncased`
  - Attribution and attention analysis only
  
- **RoBERTa** (Facebook)
  - `roberta-base`, `roberta-large`
  - Attribution and attention analysis only

## Adding New Models

The library automatically detects model architecture and adapts accordingly. To add support for a new model:

### 1. Check Model Compatibility
```python
from interpboard.core.utils import load_model_and_tokenizer

# Try loading your model
model, tokenizer = load_model_and_tokenizer("your-model-name")
print(f"Model type: {type(model)}")
print(f"Architecture: {model.config.model_type}")
```

### 2. Test Core Features
```python
from interpboard.dashboards import create_unified_dashboard

# Test dashboard creation
attribution_dashboard, ablation_dashboard = create_unified_dashboard("your-model-name")

# Test basic analysis
result = attribution_dashboard.analyze("Test sentence", methods=["integrated_gradients"])
```

### 3. Model-Specific Configurations
If needed, create custom configurations:

```python
from interpboard.core.config import Config

config = Config()
config.model.max_length = 2048  # Adjust for your model
config.patching.batch_size = 8   # Adjust for memory constraints

# Use with dashboards
attribution_dashboard, ablation_dashboard = create_unified_dashboard(
    "your-model-name", 
    config=config
)
```

## Known Limitations

### Memory Requirements
- **Large models** (>7B parameters) require significant GPU memory
- Use smaller batch sizes or model sharding for very large models
- Consider using `device_map="auto"` for automatic device placement

### Architecture-Specific Notes
- **Encoder-only models**: No causal generation, limited patching support
- **Encoder-decoder models**: Separate analysis for encoder and decoder
- **Mixture of Experts**: May require special handling for routing analysis

## Model-Specific Examples

### GPT-2 Large
```python
attribution_dashboard, ablation_dashboard = create_unified_dashboard("gpt2-large")
result = ablation_dashboard.analyze(
    "The capital of France is",
    analyses=["logit_lens", "head_ablation"],
    layers=list(range(30, 36))  # Focus on final layers
)
```

### LLaMA-2 7B
```python
# Requires authentication
from huggingface_hub import login
login()

attribution_dashboard, ablation_dashboard = create_unified_dashboard("meta-llama/Llama-2-7b-hf")
result = attribution_dashboard.analyze(
    "The Eiffel Tower is located in",
    methods=["integrated_gradients", "attention_rollout"]
)
```

### Mistral 7B
```python
attribution_dashboard, ablation_dashboard = create_unified_dashboard("mistralai/Mistral-7B-v0.1")

# Mistral has different layer structure
result = ablation_dashboard.analyze(
    "Paris is the capital of",
    layers=list(range(28, 32)),  # Final layers
    analyses=["patching", "logit_lens"]
)
```

## Performance Considerations

### Model Size Guidelines
- **Small models** (<1B): All features work smoothly
- **Medium models** (1B-7B): May need batch size adjustments
- **Large models** (7B+): Requires GPU with sufficient memory

### Optimization Tips
```python
# For large models, use smaller analysis windows
config = Config()
config.patching.max_components_per_batch = 4
config.attribution.batch_size = 8
config.visualization.max_tokens_display = 50
```

### Multi-GPU Support
```python
# For very large models
model, tokenizer = load_model_and_tokenizer(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",  # Automatic device placement
    torch_dtype=torch.float16  # Use half precision
)
```

## 🆕 Requesting New Model Support

If you encounter issues with a specific model, please:

1. Open an issue on GitHub with:
   - Model name and source
   - Error messages
   - Expected behavior

2. Include a minimal reproduction example:
```python
from interpboard.dashboards import create_unified_dashboard

try:
    attribution_dashboard, ablation_dashboard = create_unified_dashboard("model-name")
    result = attribution_dashboard.analyze("test text")
except Exception as e:
    print(f"Error: {e}")
```

The library is designed to be extensible, and we welcome contributions for new model architectures!