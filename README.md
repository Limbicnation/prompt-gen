# Prompt Generator

Generate high-quality Stable Diffusion/Flux prompts using LLMs via Ollama.

**Default model:** Qwen3-8B (~7s per prompt, 5GB)

## Quick Start

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull qwen3:8b

# Clone and install
git clone https://github.com/Limbicnation/prompt-gen.git
cd prompt-gen
pip install -r requirements.txt

# Launch Web UI
python app.py
# → Opens http://localhost:7860
```

## Usage

### Web UI (Recommended)

```bash
python app.py
```

Features:

- Style dropdown with 7 presets
- Temperature & Top-P sliders
- Emphasis and Mood inputs
- Include reasoning toggle

### Command Line

```bash
# Basic usage
python qwen_generator.py "a cyberpunk city at night" --style cyberpunk

# With advanced options
python qwen_generator.py "ancient ruins" --style fantasy --temperature 0.8 --emphasis "lighting"

# Generate multiple variations
python qwen_generator.py "ancient ruins" --style fantasy --variations 3

# Save to file
python qwen_generator.py "portrait of a wizard" --output prompts.json

# List available styles
python qwen_generator.py --list-styles
```

### Python API

```python
from qwen_generator import QwenGenerator

generator = QwenGenerator()
prompt = generator.generate_prompt(
    "a mystical forest at twilight",
    style="fantasy"
)
print(prompt)
```

## Available Styles

| Style | Description |
|-------|-------------|
| `cinematic` | Dramatic lighting and composition |
| `anime` | Vibrant anime-style illustration |
| `photorealistic` | High-detail realistic images |
| `fantasy` | Magical elements and themes |
| `abstract` | Artistic abstract compositions |
| `cyberpunk` | Neon lights, technology, urban dystopia |
| `sci-fi` | Futuristic technology scenes |

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai) with `qwen3:8b` model
- No GPU required (Ollama handles inference)

### Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model (~5GB)
ollama pull qwen3:8b

# Start the server (if not auto-started)
ollama serve
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--style` | cinematic | Style preset to use |
| `--emphasis` | - | Focus area for the prompt |
| `--mood` | - | Mood/atmosphere for the prompt |
| `--temperature` | 0.7 | Generation creativity (0.1-1.0) |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--variations` | 1 | Number of prompt variations |
| `--output` | - | Save results to JSON file |
| `--model` | qwen3:8b | Ollama model to use |
| `--timeout` | 120 | Generation timeout (seconds) |
| `--raw` | false | Keep reasoning in output |
| `--list-styles` | - | List available styles |

## Example Output

**Input:** `"a cyberpunk street market at night" --style cyberpunk`

**Output:**

```
A hyper-detailed cyberpunk street market at night, rendered in ultra-high 
resolution (8K), with a dynamic composition featuring a winding, rain-slicked 
pavement leading through a labyrinth of neon-lit stalls. Towering holographic 
advertisements project iridescent gradients of electric blue, magenta, and toxic 
green, casting kaleidoscopic reflections on wet concrete...
```

## Comparison with Other Models

This project evaluated multiple models. Qwen3-8B was chosen for:

| Factor | Qwen3-8B | DeepSeek-R1-8B |
|--------|----------|----------------|
| Speed | ~7s per prompt | ~5min per prompt |
| Output Quality | Detailed, specific | Shorter, generic |
| Model Size | ~5GB | ~15GB |
| Deployment | Simple (Ollama) | Complex (HuggingFace) |

See `evaluation_results/` for detailed comparison data.

## License

Apache License 2.0
