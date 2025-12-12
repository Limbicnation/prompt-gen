# Prompt Generator

A Python tool that uses Qwen3-8B via Ollama to generate high-quality prompts for Stable Diffusion and Flux.

## Project Overview

* **Core Technology:** Qwen3-8B via Ollama
* **Purpose:** Generate detailed, style-specific prompts for generative AI art
* **Key Features:**
  * Fast generation (~7s per prompt)
  * 7 style presets (cinematic, anime, fantasy, cyberpunk, etc.)
  * Simple Ollama integration (no GPU management needed)
  * Automatic reasoning extraction (strips "Thinking..." blocks)

## Key Files

* **`qwen_generator.py`**: Main application - CLI and API for prompt generation
* **`evaluate_models.py`**: Model comparison tool (Qwen3 vs DeepSeek)
* **`data/style_templates.json`**: Customizable style definitions

## Usage

```bash
# Basic usage
python qwen_generator.py "a mystical forest" --style fantasy

# List styles
python qwen_generator.py --list-styles

# Multiple variations
python qwen_generator.py "cyberpunk city" --variations 3 --output prompts.json
```

## Requirements

* Python 3.8+
* Ollama with `qwen3:8b` model

```bash
ollama pull qwen3:8b
```

## Development Notes

* **Style System:** Templates in `data/style_templates.json`
* **Output Cleaning:** `extract_final_prompt()` removes reasoning blocks
* **Timeout:** Default 120s, configurable via `--timeout`
