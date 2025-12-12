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
from typing import Any, Dict, List, Optional

# Optional imports for enhanced functionality
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from jinja2 import Template, Environment, BaseLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import ollama
    OLLAMA_API_AVAILABLE = True
except ImportError:
    OLLAMA_API_AVAILABLE = False


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
    
    # Default style templates (fallback when YAML not available)
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
        extract_prompt: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize the Qwen prompt generator.
        
        Args:
            model_name: Ollama model to use (default: qwen3:8b)
            timeout: Generation timeout in seconds
            extract_prompt: If True, strip thinking/reasoning from output
            temperature: Generation temperature (0.0-1.0)
            top_p: Top-p sampling parameter (0.0-1.0)
        """
        self.model_name = model_name
        self.timeout = timeout
        self.extract_prompt = extract_prompt
        self.temperature = temperature
        self.top_p = top_p
        
        # Load style templates (YAML with Jinja2, or fallback to defaults)
        self.style_templates = self._load_templates()
        
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
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load style templates from YAML file or use defaults."""
        template_path = Path(__file__).parent / "config" / "templates.yaml"
        
        if YAML_AVAILABLE and template_path.exists():
            try:
                with open(template_path, 'r') as f:
                    templates = yaml.safe_load(f)
                    return templates if templates else self.DEFAULT_STYLES
            except Exception as e:
                print(f"Warning: Failed to load templates.yaml: {e}")
        
        # Also try data/style_templates.json for backwards compatibility
        json_path = Path(__file__).parent / "data" / "style_templates.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return self.DEFAULT_STYLES
    
    def _render_template(
        self,
        style: str,
        description: str,
        emphasis: Optional[str] = None,
        mood: Optional[str] = None
    ) -> str:
        """
        Render a Jinja2 template with the given variables.
        
        Args:
            style: Style template key
            description: Main image description
            emphasis: Optional emphasis/focus area
            mood: Optional mood/atmosphere
            
        Returns:
            Rendered prompt string
        """
        template_data = self.style_templates.get(style)
        
        # Handle YAML format with 'template' key
        if isinstance(template_data, dict) and 'template' in template_data:
            template_str = template_data['template']
        elif isinstance(template_data, str):
            # Fallback: Simple string template (legacy format)
            template_str = f"""Write a detailed Stable Diffusion prompt for: {{{{ description }}}}

Style reference: {template_data}

Include specific details about:
- Composition
- Lighting
- Colors
- Atmosphere
- Technical qualities

Format the response as a single, detailed prompt."""
        else:
            # Ultimate fallback
            template_str = f"""Write a detailed Stable Diffusion prompt for: {{{{ description }}}}

Style reference: Create a cinematic scene with dramatic lighting and composition

Include specific details about:
- Composition
- Lighting
- Colors
- Atmosphere
- Technical qualities

Format the response as a single, detailed prompt."""
        
        # Render with Jinja2 if available
        if JINJA2_AVAILABLE:
            env = Environment(loader=BaseLoader())
            template = env.from_string(template_str)
            return template.render(
                description=description,
                emphasis=emphasis,
                mood=mood
            )
        else:
            # Simple string substitution fallback
            result = template_str.replace("{{ description }}", description)
            if emphasis:
                result = result.replace("{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}", f"Focus particularly on: {emphasis}")
            else:
                result = re.sub(r'\{% if emphasis %\}.*?\{% endif %\}', '', result)
            if mood:
                result = result.replace("{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}", f"Mood/Atmosphere: {mood}")
            else:
                result = re.sub(r'\{% if mood %\}.*?\{% endif %\}', '', result)
            return result
    
    def generate_prompt(
        self,
        description: str,
        style: str = "cinematic",
        emphasis: Optional[str] = None,
        mood: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extract_prompt: Optional[bool] = None
    ) -> str:
        """
        Generate a detailed image generation prompt.
        
        Args:
            description: Brief description to expand
            style: Style template to use
            emphasis: Optional focus area
            mood: Optional mood/atmosphere
            temperature: Generation temperature (overrides instance default)
            top_p: Top-p sampling (overrides instance default)
            extract_prompt: Whether to extract prompt (overrides instance default)
            
        Returns:
            Detailed prompt for image generation
        """
        prompt = self._render_template(style, description, emphasis, mood)
        
        # Use provided params or fall back to instance defaults
        final_temp = temperature if temperature is not None else self.temperature
        final_top_p = top_p if top_p is not None else self.top_p
        should_extract = extract_prompt if extract_prompt is not None else self.extract_prompt
        
        # Use Ollama Python API if available (supports temperature/top_p)
        if OLLAMA_API_AVAILABLE:
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": final_temp,
                        "top_p": final_top_p
                    }
                )
                output = response.get('response', '').strip()
                
                if should_extract:
                    output = extract_final_prompt(output)
                
                return output
            except Exception as e:
                print(f"Ollama API error: {e}, falling back to subprocess")
        
        # Fallback to subprocess (no temperature/top_p control)
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
            
            if should_extract:
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
        style: str = "cinematic",
        emphasis: Optional[str] = None,
        mood: Optional[str] = None
    ) -> List[str]:
        """
        Generate multiple prompt variations.
        
        Args:
            description: Brief description to expand
            num_variations: Number of variations to generate
            style: Style template to use
            emphasis: Optional focus area
            mood: Optional mood/atmosphere
            
        Returns:
            List of generated prompts
        """
        variations = []
        for i in range(num_variations):
            print(f"  Generating variation {i+1}/{num_variations}...")
            prompt = self.generate_prompt(description, style, emphasis, mood)
            if prompt:
                variations.append(prompt)
        return variations
    
    def get_style_info(self) -> Dict[str, Dict[str, str]]:
        """
        Get style names and descriptions for UI display.
        
        Returns:
            Dict mapping style keys to {name, description}
        """
        result = {}
        for key, value in self.style_templates.items():
            if isinstance(value, dict):
                result[key] = {
                    "name": value.get("name", key.title()),
                    "description": value.get("description", "")
                }
            else:
                result[key] = {
                    "name": key.title(),
                    "description": value if isinstance(value, str) else ""
                }
        return result


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
    parser.add_argument("description", type=str, nargs='?', help="Image description to expand")
    parser.add_argument("--style", type=str, default="cinematic",
                        choices=list(QwenGenerator.DEFAULT_STYLES.keys()),
                        help="Style preset to use (default: cinematic)")
    parser.add_argument("--emphasis", type=str, help="Focus area for the prompt")
    parser.add_argument("--mood", type=str, help="Mood/atmosphere for the prompt")
    parser.add_argument("--variations", type=int, default=1,
                        help="Number of variations to generate (default: 1)")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--model", type=str, default="qwen3:8b",
                        help="Ollama model to use (default: qwen3:8b)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Generation timeout in seconds (default: 120)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature 0.0-1.0 (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling 0.0-1.0 (default: 0.9)")
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
    
    if not args.description:
        parser.error("description is required (unless using --list-styles)")
    
    print(f"Qwen Prompt Generator - Generating prompts for: {args.description}")
    print(f"Style: {args.style}, Model: {args.model}")
    if OLLAMA_API_AVAILABLE:
        print(f"Temperature: {args.temperature}, Top-P: {args.top_p}")
    
    generator = QwenGenerator(
        model_name=args.model,
        timeout=args.timeout,
        extract_prompt=not args.raw,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    if args.variations > 1:
        prompts = generator.generate_variations(
            args.description,
            args.variations,
            args.style,
            args.emphasis,
            args.mood
        )
        result = {
            "description": args.description,
            "style": args.style,
            "model": args.model,
            "prompts": prompts
        }
    else:
        prompt = generator.generate_prompt(
            args.description, 
            args.style,
            args.emphasis,
            args.mood
        )
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
