"""
Tests for qwen_generator module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_generator import extract_final_prompt, QwenGenerator, GenerationError


class TestExtractFinalPrompt:
    """Tests for the extract_final_prompt function."""
    
    def test_removes_thinking_blocks(self):
        """Test that Qwen3's thinking blocks are removed."""
        text = "Thinking...\nsome thought\n...done thinking.\nActual prompt"
        assert extract_final_prompt(text) == "Actual prompt"
    
    def test_removes_prompt_prefix(self):
        """Test that **Prompt:** prefix is removed."""
        assert extract_final_prompt("**Prompt:** my prompt") == "my prompt"
    
    def test_removes_stable_diffusion_prompt_prefix(self):
        """Test that **Stable Diffusion Prompt:** prefix is removed."""
        assert extract_final_prompt("**Stable Diffusion Prompt:** my prompt") == "my prompt"
    
    def test_strips_quotes(self):
        """Test that surrounding quotes are stripped."""
        assert extract_final_prompt('"quoted prompt"') == "quoted prompt"
    
    def test_handles_empty_string(self):
        """Test empty string handling."""
        assert extract_final_prompt("") == ""
    
    def test_handles_none(self):
        """Test None handling."""
        assert extract_final_prompt(None) is None
    
    def test_preserves_normal_text(self):
        """Test that normal text without patterns is preserved."""
        text = "A beautiful sunset over the ocean"
        assert extract_final_prompt(text) == text


class TestQwenGeneratorValidation:
    """Tests for QwenGenerator input validation."""
    
    @pytest.fixture
    def mock_generator(self):
        """Create a generator without calling __init__."""
        gen = QwenGenerator.__new__(QwenGenerator)
        gen.style_templates = {
            "cinematic": "test template",
            "anime": "anime template"
        }
        gen.temperature = 0.7
        gen.top_p = 0.9
        gen.extract_prompt = True
        gen.timeout = 120
        gen.model_name = "qwen3:8b"
        return gen
    
    def test_rejects_empty_description(self, mock_generator):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            mock_generator.generate_prompt("")
    
    def test_rejects_whitespace_only_description(self, mock_generator):
        """Test that whitespace-only description raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            mock_generator.generate_prompt("   ")
    
    def test_rejects_too_long_description(self, mock_generator):
        """Test that overly long description raises ValueError."""
        long_desc = "x" * 10001
        with pytest.raises(ValueError, match="too long"):
            mock_generator.generate_prompt(long_desc)
    
    def test_rejects_invalid_style(self, mock_generator):
        """Test that invalid style raises ValueError."""
        with pytest.raises(ValueError, match="Unknown style"):
            mock_generator.generate_prompt("test", style="nonexistent")
    
    def test_rejects_invalid_temperature(self, mock_generator):
        """Test that out-of-range temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be"):
            mock_generator.generate_prompt("test", style="cinematic", temperature=3.0)
    
    def test_rejects_negative_temperature(self, mock_generator):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be"):
            mock_generator.generate_prompt("test", style="cinematic", temperature=-0.5)
    
    def test_rejects_invalid_top_p(self, mock_generator):
        """Test that out-of-range top_p raises ValueError."""
        with pytest.raises(ValueError, match="Top-p must be"):
            mock_generator.generate_prompt("test", style="cinematic", top_p=1.5)


class TestGenerationError:
    """Tests for the GenerationError exception."""
    
    def test_is_exception(self):
        """Test that GenerationError is an Exception."""
        assert issubclass(GenerationError, Exception)
    
    def test_can_be_raised(self):
        """Test that GenerationError can be raised and caught."""
        with pytest.raises(GenerationError, match="test error"):
            raise GenerationError("test error")
    
    def test_has_message(self):
        """Test that GenerationError stores message."""
        try:
            raise GenerationError("my error message")
        except GenerationError as e:
            assert str(e) == "my error message"
