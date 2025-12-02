# DeepSeek Prompt Generator

This project is a Python-based tool that utilizes the `DeepSeek-R1-Distill-Llama-8B` model to generate high-quality prompts for image generation models like Stable Diffusion and Flux. It offers flexible usage modes to accommodate different hardware capabilities, including memory-optimized settings for lower VRAM environments.

## Project Overview

*   **Core Technology:** DeepSeek-R1-Distill-Llama-8B (LLM).
*   **Purpose:** Automates the creation of detailed, style-specific prompts for generative AI art.
*   **Key Features:**
    *   **Dual Modes:** Standard (8GB+ VRAM) and Memory-Optimized (4GB+ VRAM).
    *   **Model Management:** Supports both HuggingFace Hub downloads and offline local models.
    *   **Style System:** Template-based styling (e.g., cinematic, anime, fantasy) customizable via JSON.
    *   **Quantization:** Automatic 4-bit quantization support via `bitsandbytes` for efficiency.

## Key Files & Structure

*   **`deepseek_generator.py`**: The core application logic. Contains the `DeepSeekGenerator` class which handles model loading, memory management, and prompt generation.
*   **`download_model.py`**: Utility script to download and cache models locally for offline use.
*   **`verify_installation.py`**: Diagnostic script to ensure all dependencies and CUDA drivers are correctly configured.
*   **`create_directories.sh`**: Helper script to initialize the project structure (`data/`, `models/`) and default style templates.
*   **`environment.yml`**: Conda environment specification (Recommended installation method).
*   **`requirements.txt`**: Pip requirements file (Alternative installation method).
*   **`data/style_templates.json`**: JSON file defining the prompt templates for various artistic styles.

## Installation & Setup

**Recommended: Conda Environment**
This ensures all CUDA and PyTorch dependencies are compatible.

```bash
conda env create -f environment.yml
conda activate deepseek-env
python verify_installation.py
```

**Alternative: Pip**
*Note: Requires manual management of CUDA libraries.*

```bash
pip install -r requirements.txt
```

## Usage Guide

### Basic Generation
Generate a prompt based on a simple input and a style.

```bash
python deepseek_generator.py "a mystical forest" --style fantasy
```

### Memory Optimized (Low VRAM)
Use the `--optimize` flag to enable 4-bit quantization and aggressive memory management. Essential for 4GB-8GB VRAM cards.

```bash
python deepseek_generator.py "a mystical forest" --style fantasy --optimize
```

### Advanced Configuration
Full control over generation parameters.

```bash
python deepseek_generator.py "a futuristic city" \
    --style cyberpunk \
    --optimize \
    --device cuda \
    --max-length 2048 \
    --temperature 0.7 \
    --variations 2 \
    --output prompts.json
```

### Offline / Local Models
1.  **Download:** `python download_model.py --output-dir ./local_model_dir`
2.  **Run:** `python deepseek_generator.py "prompt" --model-name ./local_model_dir --optimize`

## Development Conventions

*   **Style:** The codebase follows standard Python practices (`black` formatting is suggested).
*   **Error Handling:** The `DeepSeekGenerator` class is designed to gracefully handle model loading failures and fall back to different configurations (e.g., disabling quantization if unavailable).
*   **Memory Safety:** Explicit calls to `torch.cuda.empty_cache()` are used to prevent OOM errors during iterative generation.
*   **Reasoning:** The model is configured to enforce a `<think>` token pattern, useful for structured outputs like math or complex reasoning, though primarily used here for creative detailing.

## Troubleshooting

*   **OOM Errors:** Use `--optimize`, reduce `--max-length`, or decrease `--variations`.
*   **Model Not Found:** Ensure local paths are absolute or correctly relative. Use `--skip-local-check` if working on network drives.
*   **Dependencies:** If you encounter `bitsandbytes` or `torch` errors, prefer the Conda installation method to resolve binary mismatches.
