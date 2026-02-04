#!/usr/bin/env python3
"""
Test script for LLM-based keyword extraction.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from story_reader.utils.llm_nlp import LLMKeywordExtractor


def test_llm_keyword_extraction():
    """Test LLM keyword extraction with sample text."""
    
    # Test cases
    test_cases = [
        "The old lighthouse stood proudly against the stormy sea",
        "A cat sleeping on a windowsill in the afternoon sun",
        "Children playing in a sunlit meadow with flowers",
        "The bustling city street was filled with people and cars"
    ]
    
    print("Testing LLM-based keyword extraction...")
    print("=" * 50)
    
    # Test with Phi-2 model
    try:
        extractor = LLMKeywordExtractor(
            model_name="microsoft/phi-2",
            device="cpu",
            quantize=True
        )
        
        for i, text in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Input: '{text}'")
            
            # Generate a sample SD prompt
            sd_prompt = f"photojournalistic style, documentary photography, capturing: {text}"
            
            try:
                keywords = extractor.extract_keywords(text, sd_prompt, max_keywords=3)
                print(f"Keywords: {keywords}")
            except Exception as e:
                print(f"Error: {e}")
                # Try simple extraction as fallback
                print("Falling back to simple extraction...")
                keywords = extractor._simple_keyword_extraction(text, 3)
                print(f"Simple keywords: {keywords}")
    
    except Exception as e:
        print(f"Failed to load LLM model: {e}")
        print("Testing simple keyword extraction instead...")
        
        # Test simple extraction
        extractor = LLMKeywordExtractor(
            model_name="microsoft/phi-2",
            device="cpu",
            quantize=False
        )
        
        for i, text in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Input: '{text}'")
            keywords = extractor._simple_keyword_extraction(text, 3)
            print(f"Simple keywords: {keywords}")


if __name__ == "__main__":
    test_llm_keyword_extraction()