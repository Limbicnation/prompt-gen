# Prompt Generator Project Evaluation

**Date:** 2025-12-12
**Status:** Validation Required Before Further Development
**Constraint:** Open-source models only

---

## Project Summary

A Python-based prompt generator using DeepSeek-R1-Distill-Llama-8B for creating Stable Diffusion/Flux prompts.

**Current state:**
- CLI tool with 475 lines of working code
- Memory optimizations for 4GB+ VRAM
- 7 style presets (cinematic, anime, fantasy, etc.)
- Local model caching

**Target users:** Experienced developers comfortable with CLI and WebUI

**Value proposition:**
1. Local/offline capability (privacy, cost, no API dependency)
2. Style consistency via templates
3. Integration point for larger workflows (WIP)

---

## The Core Question

Before investing time in Gradio, HuggingFace Spaces, or multi-model support:

**Which open-source model produces the best prompts for the resource cost?**

### Concerns to Address

| Concern | Risk |
|---------|------|
| Time investment | Gradio/HF deployment takes 10+ hours for unproven value |
| Redundancy | Other open-source tools may work just as well |
| Value uncertainty | Output quality may not justify running a local LLM |
| Model choice | DeepSeek-R1 8B may be overkill if smaller models perform similarly |

---

## Models to Compare

All open-source, all runnable locally:

| Model | Parameters | VRAM | Key Feature |
|-------|------------|------|-------------|
| DeepSeek-R1-Distill-Llama-8B | 8B | ~6-8GB (4-bit) | `<think>` reasoning |
| Qwen3-4B-Instruct-2507-FP8 | 4B | ~4-5GB (FP8) | Native FP8, 262K context |
| Manual baseline | - | - | Your own prompt writing |

### Why These Models?

**DeepSeek-R1-8B (current):**
- Already implemented in your tool
- Reasoning via `<think>` blocks may produce structured prompts
- Larger = potentially more capable

**Qwen3-4B-FP8 (candidate):**
- Half the size, faster inference
- Native FP8 quantization (simpler deployment)
- Strong creative writing benchmarks (83.5 WritingBench)
- No `<think>` reasoning - direct output
- Open-source: Apache 2.0 license

**Manual baseline:**
- Tests whether any model adds value over writing prompts yourself
- Zero compute cost reference point

---

## Validation Plan

**Total time required: ~60 minutes**

### Step 1: Select Test Concepts (5 min)

Choose 3 prompts you'd actually use in your workflow:

```
1. "a cyberpunk street market at night"
2. "portrait of an elderly wizard"
3. "abandoned space station interior"
```

*(Replace with concepts relevant to your actual work)*

### Step 2: Write Manual Prompts (10 min)

For each concept, write a detailed SD/Flux prompt yourself. Spend 2-3 minutes per prompt. This is your baseline.

### Step 3: Generate with DeepSeek-R1 (10 min)

```bash
python deepseek_generator.py "a cyberpunk street market at night" --style cyberpunk --output deepseek_test1.json
python deepseek_generator.py "portrait of an elderly wizard" --style fantasy --output deepseek_test2.json
python deepseek_generator.py "abandoned space station interior" --style sci-fi --output deepseek_test3.json
```

### Step 4: Generate with Qwen3-4B (15 min)

Option A: Quick test via Ollama
```bash
ollama pull qwen3:4b
ollama run qwen3:4b "Write a detailed Stable Diffusion prompt for: a cyberpunk street market at night. Include composition, lighting, colors, atmosphere, and technical qualities."
```

Option B: Add Qwen3 support to your tool (future, if validated)

### Step 5: Generate Images (15 min)

- Run all 9 prompts through your image generator (SD, Flux, ComfyUI)
- Label outputs A/B/C without noting which source produced them
- Use consistent settings (same seed, steps, CFG if possible)

### Step 6: Blind Evaluation (5 min)

Answer these questions:

1. Which images do you prefer for each concept?
2. Is there a consistent winner across all 3?
3. Can you tell which prompts came from which source?
4. Is DeepSeek's `<think>` reasoning adding value over Qwen3's direct output?
5. Is either model significantly better than your manual prompts?
6. Is the quality difference worth the compute cost?

---

## Decision Framework

### Option A: Continue with DeepSeek + Add Gradio

**Choose if:**
- DeepSeek-R1 consistently produces better images than Qwen3 and manual
- The `<think>` reasoning creates prompts you wouldn't write yourself
- Style templates provide measurable consistency

**Next steps:**
1. Add minimal Gradio interface (~100 lines)
2. Deploy to HuggingFace Spaces (GPU tier)
3. Keep DeepSeek-R1 as primary model

### Option B: Switch to Qwen3-4B + Add Gradio

**Choose if:**
- Qwen3 produces similar or better results with half the resources
- Faster inference matters for your workflow
- Native FP8 simplifies deployment

**Next steps:**
1. Refactor generator to support Qwen3-4B
2. Remove DeepSeek-specific `<think>` handling
3. Add Gradio interface
4. Deploy to HuggingFace Spaces (lower GPU requirements)

### Option C: Multi-Model Support

**Choose if:**
- Both models have different strengths
- DeepSeek better for complex scenes, Qwen3 for simpler prompts
- Worth the added complexity

**Next steps:**
1. Abstract model loading to support both
2. Add model selector to CLI and Gradio
3. Document when to use each model

### Option D: Maintain CLI-Only (No Gradio)

**Choose if:**
- Results are good but not worth Gradio investment
- CLI covers your personal workflow needs
- Limited time, other priorities

**Action:**
- Keep the CLI tool for personal use
- Don't invest in Gradio/HF deployment
- Optionally add Qwen3 support for flexibility

### Option E: Discontinue

**Choose if:**
- Manual prompts consistently produce equal or better results
- Neither model adds enough value to justify compute cost
- Other tools (ComfyUI prompt nodes) cover your needs

**Action:**
- Archive the repository
- Document learnings
- Use simpler approaches instead

---

## After Validation: Gradio Implementation (If Proceeding)

### Technical Feasibility

| Aspect | Assessment |
|--------|------------|
| Gradio integration | Simple - wrap existing generator class |
| Local deployment | Works with current code |
| HF Spaces (DeepSeek) | Requires T4/A10 GPU (~$0.60/hr) |
| HF Spaces (Qwen3) | Could work on smaller GPU or ZeroGPU |
| Development time | ~4-6 hours for basic interface |

### Minimal Gradio Interface

```python
# ~80-100 lines to add:
# - Text input for description
# - Dropdown for style selection
# - Model selector (if multi-model)
# - Sliders for temperature/top_p
# - Output textbox for generated prompt
# - Copy button
```

### HuggingFace Spaces Options

| Option | Cost | Speed | Notes |
|--------|------|-------|-------|
| GPU Space (T4) | ~$0.60/hr | Fast | Required for DeepSeek-8B |
| GPU Space (T4) | ~$0.60/hr | Faster | Qwen3-4B runs better |
| ZeroGPU | Free | Queue-based | May work for Qwen3-4B |
| CPU-only | Free | Very slow | Not recommended |

---

## Open Questions

1. Does `<think>` reasoning actually improve prompt quality, or just add latency?
2. Is Qwen3-4B's creative writing strength relevant for SD prompts?
3. Would a fine-tuned prompt-specific model outperform both?
4. Is there value in supporting multiple models, or is simplicity better?

---

## Next Action

**Run the validation test before any further development.**

### Validation Results

| Concept | Manual | DeepSeek-R1 | Qwen3-4B | Winner | Notes |
|---------|--------|-------------|----------|--------|-------|
| Cyberpunk market | | | | | |
| Elderly wizard | | | | | |
| Space station | | | | | |

**Overall winner:** ____________

**Decision:** [ ] A: DeepSeek + Gradio [ ] B: Qwen3 + Gradio [ ] C: Multi-model [ ] D: CLI-only [ ] E: Discontinue

**Reasoning:**

**Date validated:** ____________
