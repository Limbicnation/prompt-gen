"""
Tests for style_presets module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from style_presets import (
    StyleMode,
    StyleKeywords,
    StylePreset,
    get_style_keywords,
    get_available_styles,
    STYLE_DEFINITIONS
)


class TestStyleMode:
    """Tests for StyleMode enum."""
    
    def test_cinematic_value(self):
        assert StyleMode.CINEMATIC.value == "cinematic"
    
    def test_still_image_value(self):
        assert StyleMode.STILL_IMAGE.value == "still_image"


class TestStyleKeywords:
    """Tests for StyleKeywords dataclass."""
    
    def test_to_prompt_string(self):
        keywords = StyleKeywords(
            primary=["one", "two"],
            lighting=["three"],
            technical=[],
            composition=[],
            texture=[]
        )
        assert keywords.to_prompt_string() == "one, two, three"
    
    def test_to_list(self):
        keywords = StyleKeywords(
            primary=["a"],
            lighting=["b"],
            technical=["c"],
            composition=["d"],
            texture=["e"]
        )
        assert keywords.to_list() == ["a", "b", "c", "d", "e"]
    
    def test_empty_keywords(self):
        keywords = StyleKeywords()
        assert keywords.to_list() == []
        assert keywords.to_prompt_string() == ""


class TestGetStyleKeywords:
    """Tests for get_style_keywords function."""
    
    def test_cinematic_returns_keywords(self):
        keywords = get_style_keywords("cinematic")
        assert isinstance(keywords, StyleKeywords)
        assert len(keywords.primary) > 0
        assert "cinematic shot" in keywords.primary
    
    def test_still_image_returns_keywords(self):
        keywords = get_style_keywords("still_image")
        assert isinstance(keywords, StyleKeywords)
        assert "professional photography" in keywords.primary
    
    def test_case_insensitive(self):
        keywords1 = get_style_keywords("CINEMATIC")
        keywords2 = get_style_keywords("cinematic")
        assert keywords1.to_list() == keywords2.to_list()
    
    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="Unknown style"):
            get_style_keywords("nonexistent")


class TestGetAvailableStyles:
    """Tests for get_available_styles function."""
    
    def test_returns_list(self):
        styles = get_available_styles()
        assert isinstance(styles, list)
    
    def test_contains_expected_styles(self):
        styles = get_available_styles()
        assert "cinematic" in styles
        assert "still_image" in styles


class TestStylePreset:
    """Tests for StylePreset class."""
    
    @pytest.fixture
    def preset(self):
        return StylePreset()
    
    def test_get_style_choices(self):
        choices = StylePreset.get_style_choices()
        assert isinstance(choices, tuple)
        assert "cinematic" in choices
        assert "still_image" in choices
    
    def test_get_style_keywords(self, preset):
        keywords = preset.get_style_keywords("cinematic")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
    
    def test_get_style_prompt(self, preset):
        prompt = preset.get_style_prompt("cinematic")
        assert isinstance(prompt, str)
        assert "cinematic shot" in prompt
    
    def test_get_style_prompt_with_emphasis(self, preset):
        prompt = preset.get_style_prompt("cinematic", emphasis="lighting")
        assert "emphasis on lighting" in prompt
    
    def test_get_style_prompt_without_technical(self, preset):
        prompt = preset.get_style_prompt("still_image", include_technical=False)
        assert "ISO" not in prompt
    
    def test_apply_to_prompt_append(self, preset):
        result = preset.apply_to_prompt("a cat", "cinematic")
        assert result.startswith("a cat,")
        assert "cinematic shot" in result
    
    def test_apply_to_prompt_prepend(self, preset):
        result = preset.apply_to_prompt("a cat", "cinematic", position="prepend")
        assert result.endswith("a cat")
        assert "cinematic shot" in result
    
    def test_get_style_info(self, preset):
        info = preset.get_style_info()
        assert "cinematic" in info
        assert "still_image" in info
        assert info["cinematic"]["name"] == "Cinematic"
        assert "keywords_count" in info["cinematic"]
