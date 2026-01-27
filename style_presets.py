"""
Style Presets Module
Modular style system for prompt generation.
Designed for easy integration with ComfyUI custom nodes.

Usage:
    from style_presets import StylePreset, get_style_keywords
    
    # Get style keywords
    keywords = get_style_keywords("cinematic")
    
    # Use StylePreset class (ComfyUI-compatible)
    preset = StylePreset()
    prompt_additions = preset.get_style_prompt("still_image", emphasis="macro details")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class StyleMode(Enum):
    """Enumeration of available style modes."""
    CINEMATIC = "cinematic"
    STILL_IMAGE = "still_image"


@dataclass
class StyleKeywords:
    """Container for style-specific keywords and descriptors.
    
    Attributes:
        primary: Core style keywords (always included)
        lighting: Lighting-related descriptors
        technical: Technical/camera specifications
        composition: Composition and framing keywords
        texture: Texture and detail descriptors
    """
    primary: List[str] = field(default_factory=list)
    lighting: List[str] = field(default_factory=list)
    technical: List[str] = field(default_factory=list)
    composition: List[str] = field(default_factory=list)
    texture: List[str] = field(default_factory=list)
    
    def to_prompt_string(self, separator: str = ", ") -> str:
        """Convert all keywords to a single prompt string.
        
        Args:
            separator: String to join keywords with
            
        Returns:
            Combined keyword string
        """
        all_keywords = (
            self.primary + 
            self.lighting + 
            self.technical + 
            self.composition + 
            self.texture
        )
        return separator.join(all_keywords)
    
    def to_list(self) -> List[str]:
        """Return all keywords as a flat list (ComfyUI-compatible).
        
        Returns:
            List of all keywords
        """
        return (
            self.primary + 
            self.lighting + 
            self.technical + 
            self.composition + 
            self.texture
        )


@dataclass
class StyleDefinition:
    """Complete style definition with metadata and keywords.
    
    This is the single source of truth for all style information.
    
    Attributes:
        name: Human-readable display name
        description: Brief description for UI tooltips
        keywords: StyleKeywords instance with all keywords
    """
    name: str
    description: str
    keywords: StyleKeywords


# Style definitions - single source of truth for all style metadata
STYLE_DEFINITIONS: Dict[StyleMode, StyleDefinition] = {
    StyleMode.CINEMATIC: StyleDefinition(
        name="Cinematic",
        description="Film-like visuals with dramatic lighting and anamorphic qualities",
        keywords=StyleKeywords(
            primary=[
                "cinematic shot",
                "film grain",
                "movie still",
                "dramatic scene"
            ],
            lighting=[
                "dramatic lighting",
                "volumetric light",
                "rim lighting",
                "chiaroscuro",
                "golden hour"
            ],
            technical=[
                "anamorphic lens",
                "shallow depth of field",
                "bokeh",
                "35mm film",
                "wide aspect ratio"
            ],
            composition=[
                "rule of thirds",
                "leading lines",
                "dynamic composition",
                "cinematic framing"
            ],
            texture=[
                "rich color grading",
                "film texture",
                "atmospheric haze"
            ]
        )
    ),
    StyleMode.STILL_IMAGE: StyleDefinition(
        name="Still Image (Photography)",
        description="Sharp, realistic photography with technical camera specifications",
        keywords=StyleKeywords(
            primary=[
                "professional photography",
                "high resolution",
                "sharp focus",
                "studio quality"
            ],
            lighting=[
                "natural lighting",
                "soft diffused light",
                "studio lighting",
                "balanced exposure"
            ],
            technical=[
                "f/2.8 aperture",
                "ISO 100",
                "sharp optics",
                "full frame sensor",
                "RAW quality"
            ],
            composition=[
                "centered composition",
                "clean framing",
                "balanced layout",
                "professional angle"
            ],
            texture=[
                "realistic textures",
                "fine details",
                "crisp definition",
                "accurate colors"
            ]
        )
    )
}


def get_style_keywords(style: str) -> StyleKeywords:
    """Get keywords for a specific style.
    
    Args:
        style: Style name ("cinematic" or "still_image")
        
    Returns:
        StyleKeywords object with all style-specific keywords
        
    Raises:
        ValueError: If style is not recognized
    """
    try:
        mode = StyleMode(style.lower())
        return STYLE_DEFINITIONS[mode].keywords
    except ValueError:
        available = [s.value for s in StyleMode]
        raise ValueError(f"Unknown style '{style}'. Available: {', '.join(available)}")


def get_available_styles() -> List[str]:
    """Get list of available style names.
    
    Returns:
        List of style name strings
    """
    return [mode.value for mode in StyleMode]


class StylePreset:
    """
    ComfyUI-compatible style preset class.
    
    This class is designed to be easily portable to ComfyUI custom nodes.
    All methods return simple types (str, list, dict) for compatibility.
    
    Example ComfyUI Integration:
        class PromptStyleNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "style": (StylePreset.get_style_choices(),),
                        "base_prompt": ("STRING", {"default": ""}),
                    }
                }
            
            RETURN_TYPES = ("STRING",)
            FUNCTION = "apply_style"
            
            def apply_style(self, style, base_prompt):
                preset = StylePreset()
                styled = preset.apply_to_prompt(base_prompt, style)
                return (styled,)
    """
    
    def __init__(self):
        """Initialize the StylePreset with available styles."""
        self._styles = STYLE_DEFINITIONS
    
    @staticmethod
    def get_style_choices() -> Tuple[str, ...]:
        """Get available style choices as tuple (ComfyUI dropdown format).
        
        Returns:
            Tuple of style names for ComfyUI dropdown
        """
        return tuple(mode.value for mode in StyleMode)
    
    def get_style_keywords(self, style: str) -> List[str]:
        """Get style keywords as a list.
        
        Args:
            style: Style name
            
        Returns:
            List of keyword strings
        """
        keywords = get_style_keywords(style)
        return keywords.to_list()
    
    def get_style_prompt(
        self,
        style: str,
        emphasis: Optional[str] = None,
        include_technical: bool = True
    ) -> str:
        """Get a formatted style prompt string.
        
        Args:
            style: Style name ("cinematic" or "still_image")
            emphasis: Optional emphasis to highlight specific aspects
            include_technical: Whether to include technical camera specs
            
        Returns:
            Formatted prompt string with style keywords
        """
        keywords = get_style_keywords(style)
        
        parts = []
        parts.extend(keywords.primary)
        parts.extend(keywords.lighting)
        parts.extend(keywords.composition)
        parts.extend(keywords.texture)
        
        if include_technical:
            parts.extend(keywords.technical)
        
        if emphasis:
            parts.insert(0, f"emphasis on {emphasis}")
        
        return ", ".join(parts)
    
    def apply_to_prompt(
        self,
        base_prompt: str,
        style: str,
        position: str = "append"
    ) -> str:
        """Apply style keywords to an existing prompt.
        
        Args:
            base_prompt: The original prompt text
            style: Style to apply
            position: "append" or "prepend"
            
        Returns:
            Modified prompt with style keywords
        """
        style_prompt = self.get_style_prompt(style)
        
        if position == "prepend":
            return f"{style_prompt}, {base_prompt}"
        else:
            return f"{base_prompt}, {style_prompt}"
    
    def get_style_info(self) -> Dict[str, Dict[str, any]]:
        """Get detailed information about all styles (for UI display).
        
        Returns:
            Dict mapping style names to info dicts
        """
        return {
            mode.value: {
                "name": definition.name,
                "description": definition.description,
                "keywords_count": len(definition.keywords.to_list())
            }
            for mode, definition in self._styles.items()
        }


# For direct import convenience
__all__ = [
    "StyleMode",
    "StyleKeywords",
    "StyleDefinition",
    "StylePreset",
    "get_style_keywords",
    "get_available_styles",
    "STYLE_DEFINITIONS"
]
