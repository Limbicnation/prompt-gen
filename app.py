#!/usr/bin/env python3
"""
Gradio Web UI for Prompt Generator
A user-friendly interface for generating Stable Diffusion prompts using Qwen3-8B via Ollama
"""

import gradio as gr
from qwen_generator import QwenGenerator, OLLAMA_API_AVAILABLE

# Global generator instance (loaded once on startup for performance)
generator: QwenGenerator = None


def get_generator() -> QwenGenerator:
    """Get or create the global generator instance."""
    global generator
    if generator is None:
        generator = QwenGenerator()
    return generator


def generate_prompt(
    description: str,
    style: str,
    emphasis: str,
    mood: str,
    temperature: float,
    top_p: float,
    include_reasoning: bool
) -> str:
    """
    Generate a prompt using the QwenGenerator.
    
    Args:
        description: Image description
        style: Style template key
        emphasis: Optional focus area
        mood: Optional mood/atmosphere
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        include_reasoning: If True, keep reasoning in output
        
    Returns:
        Generated prompt string
    """
    if not description.strip():
        return "⚠️ Please enter an image description."
    
    gen = get_generator()
    
    try:
        result = gen.generate_prompt(
            description=description.strip(),
            style=style,
            emphasis=emphasis.strip() if emphasis else None,
            mood=mood.strip() if mood else None,
            temperature=temperature,
            top_p=top_p,
            extract_prompt=not include_reasoning
        )
        
        if not result:
            return "⚠️ Generation failed. Please check that Ollama is running."
        
        return result
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def build_ui() -> gr.Blocks:
    """Build and return the Gradio UI."""
    
    # Get style options from generator
    gen = get_generator()
    style_info = gen.get_style_info()
    style_choices = [(info["name"], key) for key, info in style_info.items()]
    
    with gr.Blocks(title="Prompt Generator") as app:
        gr.Markdown(
            """
            # 🎨 Prompt Generator
            Generate detailed Stable Diffusion prompts using **Qwen3-8B** via Ollama.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input Section
                description = gr.Textbox(
                    label="📝 Image Description",
                    placeholder="e.g., a mystical forest at twilight",
                    lines=3,
                    max_lines=5
                )
                
                style = gr.Dropdown(
                    label="🎭 Style",
                    choices=style_choices,
                    value="cinematic",
                    info="Select a style template"
                )
                
                with gr.Accordion("🔧 Advanced Options", open=False):
                    emphasis = gr.Textbox(
                        label="Focus Area",
                        placeholder="e.g., lighting, composition, details",
                        max_lines=1
                    )
                    
                    mood = gr.Textbox(
                        label="Mood / Atmosphere",
                        placeholder="e.g., mysterious, serene, dramatic",
                        max_lines=1
                    )
                    
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        info="Higher = more creative, lower = more focused"
                    )
                    
                    top_p = gr.Slider(
                        label="Top-P",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        info="Nucleus sampling threshold"
                    )
                    
                    include_reasoning = gr.Checkbox(
                        label="Include reasoning process",
                        value=False,
                        info="Show the model's thinking process"
                    )
                    
                    if not OLLAMA_API_AVAILABLE:
                        gr.Markdown(
                            """
                            > ⚠️ **Note:** Install `ollama` package for temperature/top_p control:  
                            > `pip install ollama`
                            """
                        )
                
                generate_btn = gr.Button("✨ Generate Prompt", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Output Section
                output = gr.Textbox(
                    label="📄 Generated Prompt",
                    lines=15,
                    max_lines=20,
                    elem_classes=["output-text"]
                )
        
        # Event handlers
        generate_btn.click(
            fn=generate_prompt,
            inputs=[description, style, emphasis, mood, temperature, top_p, include_reasoning],
            outputs=output
        )
        
        # Also trigger on Enter key in description
        description.submit(
            fn=generate_prompt,
            inputs=[description, style, emphasis, mood, temperature, top_p, include_reasoning],
            outputs=output
        )
        
        gr.Markdown(
            """
            ---
            **Tip:** Use the [Advanced Options] to fine-tune your prompts.  
            Powered by [Qwen3-8B](https://ollama.com/library/qwen3:8b) via [Ollama](https://ollama.ai)
            """
        )
    
    return app


def main():
    """Launch the Gradio UI."""
    print("🚀 Starting Prompt Generator UI...")
    print("   Loading Qwen3-8B model via Ollama...")
    
    # Pre-initialize generator
    get_generator()
    
    print("⚠️  Warning: Binding to 0.0.0.0 - accessible on your local network")
    print("   Access at: http://localhost:7860")
    
    # Build and launch
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="slate"
        )
    )


if __name__ == "__main__":
    main()
