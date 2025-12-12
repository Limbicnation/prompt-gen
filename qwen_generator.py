#!/usr/bin/env python3
"""
Qwen Prompt Generator
Generate high-quality prompts for Stable Diffusion using Qwen3-8B via Ollama
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def extract_final_prompt(text: str) -> str:
    """
    Extract the final prompt, stripping any thinking/reasoning process.
    Handles Qwen3's "Thinking..." pattern.
    """
    if not text:
        return text
    
    # Remove Qwen3 thinking blocks: "Thinking...\n...\n...done thinking.\n"
    text = re.sub(r'Thinking\.\.\.\n.*?\.\.\.done thinking\.\n*', '', text, flags=re.DOTALL)
    
    # Remove common prefixes like "**Prompt:**" or "**Stable Diffusion Prompt:**"
    text = re.sub(r'\*\*(?:Stable Diffusion )?Prompt:\*\*\s*', '', text)
    text = re.sub(r'\*\*Prompt for Image Generation:\*\*\s*', '', text)
    
    # Clean up any leading/trailing whitespace and quotes
    text = text.strip().strip('"').strip()
    
    return text


class QwenGenerator:
    """Generate image prompts using Qwen3-8B via Ollama."""
    
    # Default style templates
    DEFAULT_STYLES = {
        "cinematic": "Create a cinematic scene with dramatic lighting and composition",
        "anime": "Design an anime-style illustration with vibrant colors",
        "photorealistic": "Generate a photorealistic image with high detail",
        "fantasy": "Create a fantasy-themed illustration with magical elements",
        "abstract": "Design an abstract artistic composition",
        "cyberpunk": "Create a cyberpunk-themed image with neon lights, high technology, and urban dystopia",
        "sci-fi": "Generate a science fiction scene with futuristic technology"
    }
    
    def __init__(
        self,
        model_name: str = "qwen3:8b",
        timeout: int = 120,
        extract_prompt: bool = True
    ):
        """
        Initialize the Qwen prompt generator.
        
        Args:
            model_name: Ollama model to use (default: qwen3:8b)
            timeout: Generation timeout in seconds
            extract_prompt: If True, strip thinking/reasoning from output
        """
        self.model_name = model_name
        self.timeout = timeout
        self.extract_prompt = extract_prompt
        
        # Load style templates
        self.style_templates = self._load_style_templates()
        
        # Verify Ollama is available
        self._verify_ollama()
    
    def _verify_ollama(self):
        """Check if Ollama is installed and the model is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                print("Warning: Ollama may not be running. Start with: ollama serve")
            elif self.model_name not in result.stdout:
                print(f"Warning: Model {self.model_name} not found. Pull with: ollama pull {self.model_name}")
        except FileNotFoundError:
            raise RuntimeError("Ollama not found. Install from: https://ollama.ai")
        except subprocess.TimeoutExpired:
            print("Warning: Ollama check timed out")
    
    def _load_style_templates(self) -> Dict[str, str]:
        """Load style templates from JSON file or use defaults."""
        try:
            with open('data/style_templates.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.DEFAULT_STYLES
    
    def generate_prompt(
        self,
        description: str,
        style: str = "cinematic"
    ) -> str:
        """
        Generate a detailed image generation prompt.
        
        Args:
            description: Brief description to expand
            style: Style template to use
            
        Returns:
            Detailed prompt for image generation
        """
        style_ref = self.style_templates.get(style, self.DEFAULT_STYLES.get("cinematic"))
        
        prompt = f"""Write a detailed Stable Diffusion prompt for: {description}

Style reference: {style_ref}

Include specific details about:
- Composition
- Lighting
- Colors
- Atmosphere
- Technical qualities

Format the response as a single, detailed prompt."""

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                print(f"Error generating prompt: {result.stderr}")
                return ""
            
            output = result.stdout.strip()
            
            if self.extract_prompt:
                output = extract_final_prompt(output)
            
            return output
            
        except subprocess.TimeoutExpired:
            print(f"Generation timed out after {self.timeout}s")
            return ""
        except Exception as e:
            print(f"Error: {e}")
            return ""
    
    def generate_variations(
        self,
        description: str,
        num_variations: int = 2,
        style: str = "cinematic"
    ) -> List[str]:
        """
        Generate multiple prompt variations.
        
        Args:
            description: Brief description to expand
            num_variations: Number of variations to generate
            style: Style template to use
            
        Returns:
            List of generated prompts
        """
        variations = []
        for i in range(num_variations):
            print(f"  Generating variation {i+1}/{num_variations}...")
            prompt = self.generate_prompt(description, style)
            if prompt:
                variations.append(prompt)
        return variations


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts using Qwen3-8B via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qwen_generator.py "a mystical forest at twilight" --style fantasy
  python qwen_generator.py "cyberpunk city" --style cyberpunk --output prompts.json
  python qwen_generator.py "ancient ruins" --variations 3
        """
    )
    parser.add_argument("description", type=str, help="Image description to expand")
    parser.add_argument("--style", type=str, default="cinematic",
                        choices=list(QwenGenerator.DEFAULT_STYLES.keys()),
                        help="Style preset to use (default: cinematic)")
    parser.add_argument("--variations", type=int, default=1,
                        help="Number of variations to generate (default: 1)")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--model", type=str, default="qwen3:8b",
                        help="Ollama model to use (default: qwen3:8b)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Generation timeout in seconds (default: 120)")
    parser.add_argument("--raw", action="store_true",
                        help="Keep raw output including thinking process")
    parser.add_argument("--list-styles", action="store_true",
                        help="List available style presets and exit")
    
    args = parser.parse_args()
    
    if args.list_styles:
        print("Available styles:")
        for name, desc in QwenGenerator.DEFAULT_STYLES.items():
            print(f"  {name}: {desc}")
        return
    
    print(f"Qwen Prompt Generator - Generating prompts for: {args.description}")
    print(f"Style: {args.style}, Model: {args.model}")
    
    generator = QwenGenerator(
        model_name=args.model,
        timeout=args.timeout,
        extract_prompt=not args.raw
    )
    
    if args.variations > 1:
        prompts = generator.generate_variations(
            args.description,
            args.variations,
            args.style
        )
        result = {
            "description": args.description,
            "style": args.style,
            "model": args.model,
            "prompts": prompts
        }
    else:
        prompt = generator.generate_prompt(args.description, args.style)
        result = {
            "description": args.description,
            "style": args.style,
            "model": args.model,
            "prompt": prompt
        }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print("\nGenerated Prompt:")
        print("=" * 60)
        if args.variations > 1:
            for i, p in enumerate(result["prompts"], 1):
                print(f"\n[Variation {i}]\n{p}")
        else:
            print(result["prompt"])


if __name__ == "__main__":
    main()
