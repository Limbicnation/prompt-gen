# Prompt Generator

A Python tool that uses Qwen3-8B via Ollama to generate high-quality prompts for Stable Diffusion and Flux.

## Project Overview

* **Core Technology:** Qwen3-8B via Ollama
* **Purpose:** Generate detailed, style-specific prompts for generative AI art
* **Key Features:**
  * **Gradio Web UI** with style selector and parameter controls
  * Fast generation (~7s per prompt)
  * 7 style presets (cinematic, anime, fantasy, cyberpunk, etc.)
  * Jinja2 templating for customizable prompts
  * Temperature/Top-P controls via Ollama API
  * Automatic reasoning extraction (strips "Thinking..." blocks)

## Key Files

* **`app.py`**: Gradio web UI
* **`qwen_generator.py`**: Core generator - CLI and Python API
* **`config/templates.yaml`**: Jinja2 style templates

## Usage

```bash
# Web UI (recommended)
python app.py

# CLI
python qwen_generator.py "a mystical forest" --style fantasy

# CLI with advanced options
python qwen_generator.py "cyberpunk city" --style cyberpunk --temperature 0.8 --emphasis "neon lighting"

# List styles
python qwen_generator.py --list-styles
```

## Requirements

* Python 3.8+
* Ollama with `qwen3:8b` model

```bash
pip install -r requirements.txt
ollama pull qwen3:8b
```

## Development Notes

* **Style System:** Jinja2 templates in `config/templates.yaml`
* **Thread Safety:** `generate_prompt()` accepts params directly (stateless)
* **Output Cleaning:** `extract_final_prompt()` removes reasoning blocks
* **Timeout:** Default 120s, configurable via `--timeout`
