#!/usr/bin/env python3
"""
Model Evaluation Script
Compare DeepSeek-R1-8B vs Qwen3-8B for prompt generation quality.
"""

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Test concepts from the evaluation plan
TEST_CONCEPTS = [
    {"concept": "a cyberpunk street market at night", "style": "cyberpunk"},
    {"concept": "portrait of an elderly wizard", "style": "fantasy"},
    {"concept": "abandoned space station interior", "style": "sci-fi"},
]


def extract_final_prompt(text: str) -> str:
    """
    Extract the final prompt, stripping any thinking/reasoning process.
    Handles both Qwen3 (Thinking...) and DeepSeek (<think>) formats.
    """
    if not text:
        return text
    
    # Remove Qwen3 thinking blocks: "Thinking...\n...\n...done thinking.\n"
    text = re.sub(r'Thinking\.\.\.\n.*?\.\.\.done thinking\.\n*', '', text, flags=re.DOTALL)
    
    # Remove DeepSeek <think> blocks
    text = re.sub(r'<think>.*?</think>\n*', '', text, flags=re.DOTALL)
    
    # Remove common prefixes like "**Prompt:**" or "**Stable Diffusion Prompt:**"
    text = re.sub(r'\*\*(?:Stable Diffusion )?Prompt:\*\*\s*', '', text)
    
    # Clean up any leading/trailing whitespace and quotes
    text = text.strip().strip('"').strip()
    
    return text


def run_ollama_generation(concept: str) -> Dict:
    """Generate prompt using Qwen3 via Ollama."""
    prompt = f"""Write a detailed Stable Diffusion prompt for: {concept}

Include specific details about:
- Composition
- Lighting
- Colors
- Atmosphere
- Technical qualities

Format the response as a single, detailed prompt."""

    start_time = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen3:8b", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.time() - start_time
        raw_output = result.stdout.strip()
        return {
            "model": "qwen3:8b",
            "concept": concept,
            "prompt": extract_final_prompt(raw_output),
            "raw_output": raw_output,
            "error": result.stderr if result.returncode != 0 else None,
            "generation_time_seconds": round(elapsed, 2),
        }
    except subprocess.TimeoutExpired:
        return {
            "model": "qwen3:8b",
            "concept": concept,
            "prompt": None,
            "error": "Timeout after 120s",
            "generation_time_seconds": 120,
        }
    except Exception as e:
        return {
            "model": "qwen3:8b",
            "concept": concept,
            "prompt": None,
            "error": str(e),
            "generation_time_seconds": 0,
        }


def run_deepseek_generation(concept: str, style: str) -> Dict:
    """Generate prompt using DeepSeek via the existing generator."""
    start_time = time.time()
    try:
        result = subprocess.run(
            [
                "python", "deepseek_generator.py",
                concept,
                "--style", style,
                "--output", f"/tmp/deepseek_{hash(concept)}.json",
                "--variations", "0",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).parent),
        )
        elapsed = time.time() - start_time
        
        # Read the output file
        output_file = f"/tmp/deepseek_{hash(concept)}.json"
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
            os.remove(output_file)
            raw_output = data.get("base_prompt", "")
            return {
                "model": "deepseek-r1-8b",
                "concept": concept,
                "style": style,
                "prompt": extract_final_prompt(raw_output),
                "raw_output": raw_output,
                "error": None,
                "generation_time_seconds": round(elapsed, 2),
            }
        else:
            return {
                "model": "deepseek-r1-8b",
                "concept": concept,
                "style": style,
                "prompt": None,
                "error": result.stderr or "No output file generated",
                "generation_time_seconds": round(elapsed, 2),
            }
    except subprocess.TimeoutExpired:
        return {
            "model": "deepseek-r1-8b",
            "concept": concept,
            "style": style,
            "prompt": None,
            "error": "Timeout after 300s",
            "generation_time_seconds": 300,
        }
    except Exception as e:
        return {
            "model": "deepseek-r1-8b",
            "concept": concept,
            "style": style,
            "prompt": None,
            "error": str(e),
            "generation_time_seconds": 0,
        }


def generate_report(
    deepseek_results: List[Dict],
    qwen_results: List[Dict],
    output_dir: Path
) -> str:
    """Generate markdown comparison report."""
    report = f"""# Model Evaluation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Summary

| Model | Avg Generation Time | Success Rate |
|-------|---------------------|--------------|
| DeepSeek-R1-8B | {sum(r['generation_time_seconds'] for r in deepseek_results)/len(deepseek_results):.1f}s | {sum(1 for r in deepseek_results if r['prompt'])/len(deepseek_results)*100:.0f}% |
| Qwen3-8B | {sum(r['generation_time_seconds'] for r in qwen_results)/len(qwen_results):.1f}s | {sum(1 for r in qwen_results if r['prompt'])/len(qwen_results)*100:.0f}% |

---

## Generated Prompts

"""
    for i, concept in enumerate(TEST_CONCEPTS):
        ds = deepseek_results[i]
        qw = qwen_results[i]
        
        report += f"""### Concept {i+1}: "{concept['concept']}"

**DeepSeek-R1-8B** ({ds['generation_time_seconds']}s):
```
{ds['prompt'][:1500] if ds['prompt'] else f"ERROR: {ds['error']}"}
```

**Qwen3-8B** ({qw['generation_time_seconds']}s):
```
{qw['prompt'][:1500] if qw['prompt'] else f"ERROR: {qw['error']}"}
```

---

"""
    
    report += """## Evaluation Criteria

For each concept, rate on a scale of 1-5:

| Concept | DeepSeek Quality | Qwen3 Quality | Winner |
|---------|------------------|---------------|--------|
| Cyberpunk Market | __ | __ | __ |
| Elderly Wizard | __ | __ | __ |
| Space Station | __ | __ | __ |

**Overall Winner:** ____________

**Decision:** [ ] A: DeepSeek + Gradio [ ] B: Qwen3 + Gradio [ ] C: Multi-model [ ] D: CLI-only [ ] E: Discontinue
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate prompt generation models")
    parser.add_argument("--dry-run", action="store_true", help="Test without running models")
    parser.add_argument("--qwen-only", action="store_true", help="Only run Qwen3 tests")
    parser.add_argument("--deepseek-only", action="store_true", help="Only run DeepSeek tests")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(__file__).parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    if args.dry_run:
        print("Dry run mode - testing configuration...")
        print(f"Output directory: {output_dir}")
        print(f"Test concepts: {len(TEST_CONCEPTS)}")
        print("✓ Configuration valid")
        return
    
    print("=" * 60)
    print("Model Evaluation - Starting validation test")
    print("=" * 60)
    
    deepseek_results = []
    qwen_results = []
    
    # Run Qwen3 tests
    if not args.deepseek_only:
        print("\n[1/2] Running Qwen3-8B generation...")
        for i, test in enumerate(TEST_CONCEPTS):
            print(f"  [{i+1}/{len(TEST_CONCEPTS)}] {test['concept'][:40]}...")
            result = run_ollama_generation(test['concept'])
            qwen_results.append(result)
            if result['error']:
                print(f"    ⚠ Error: {result['error'][:50]}")
            else:
                print(f"    ✓ Generated in {result['generation_time_seconds']}s")
        
        # Save Qwen results
        with open(output_dir / "qwen3_prompts.json", 'w') as f:
            json.dump(qwen_results, f, indent=2)
        print(f"  Saved to: {output_dir}/qwen3_prompts.json")
    
    # Run DeepSeek tests  
    if not args.qwen_only:
        print("\n[2/2] Running DeepSeek-R1-8B generation...")
        for i, test in enumerate(TEST_CONCEPTS):
            print(f"  [{i+1}/{len(TEST_CONCEPTS)}] {test['concept'][:40]}...")
            result = run_deepseek_generation(test['concept'], test['style'])
            deepseek_results.append(result)
            if result['error']:
                print(f"    ⚠ Error: {result['error'][:50]}")
            else:
                print(f"    ✓ Generated in {result['generation_time_seconds']}s")
        
        # Save DeepSeek results
        with open(output_dir / "deepseek_prompts.json", 'w') as f:
            json.dump(deepseek_results, f, indent=2)
        print(f"  Saved to: {output_dir}/deepseek_prompts.json")
    
    # Generate comparison report if we have both
    if deepseek_results and qwen_results:
        print("\nGenerating comparison report...")
        report = generate_report(deepseek_results, qwen_results, output_dir)
        with open(output_dir / "comparison_report.md", 'w') as f:
            f.write(report)
        print(f"  Saved to: {output_dir}/comparison_report.md")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
