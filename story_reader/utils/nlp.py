"""
NLP utilities for keyword extraction and text processing.
"""

import re
import string
from typing import List, Set, Optional
from collections import Counter


class KeywordExtractor:
    """
    Extracts keywords from text using rule-based methods.
    
    Designed to work with both original paragraph text and Stable Diffusion prompts
    to generate optimized search queries for Pexels API.
    """
    
    # Common stop words to filter out
    STOP_WORDS: Set[str] = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall',
        'this', 'that', 'these', 'those', 'my', 'your', 'our', 'their',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
    }
    
    # Common descriptive adjectives to prioritize
    DESCRIPTIVE_ADJECTIVES: Set[str] = {
        'beautiful', 'amazing', 'stunning', 'gorgeous', 'breathtaking', 'magnificent',
        'peaceful', 'tranquil', 'serene', 'calm', 'quiet', 'still',
        'bright', 'dark', 'light', 'shadow', 'sunny', 'cloudy', 'foggy', 'misty',
        'old', 'ancient', 'modern', 'contemporary', 'vintage', 'rustic',
        'big', 'small', 'large', 'tiny', 'huge', 'massive', 'enormous',
        'colorful', 'vibrant', 'bright', 'dark', 'black', 'white', 'red', 'blue',
        'happy', 'sad', 'angry', 'joyful', 'excited', 'peaceful',
        'natural', 'artificial', 'real', 'fake', 'authentic', 'genuine'
    }
    
    # Common visual/concrete nouns to prioritize
    VISUAL_NOUNS: Set[str] = {
        'person', 'people', 'man', 'woman', 'child', 'girl', 'boy', 'family',
        'building', 'house', 'home', 'city', 'town', 'village', 'street', 'road',
        'tree', 'plant', 'flower', 'garden', 'forest', 'park', 'field', 'meadow',
        'water', 'river', 'lake', 'ocean', 'sea', 'beach', 'mountain', 'hill',
        'sky', 'cloud', 'sun', 'moon', 'star', 'night', 'day', 'morning', 'evening',
        'animal', 'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'sheep',
        'car', 'vehicle', 'transport', 'train', 'plane', 'boat', 'ship',
        'food', 'meal', 'drink', 'water', 'coffee', 'tea', 'fruit', 'vegetable'
    }
    
    def __init__(self, max_keywords: int = 3):
        """
        Initialize keyword extractor.
        
        Args:
            max_keywords: Maximum number of keywords to return (default: 3)
        """
        self.max_keywords = max_keywords
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using rule-based methods.
        
        Args:
            text: Input text to extract keywords from
            
        Returns:
            List of extracted keywords (up to max_keywords)
        """
        if not text or not text.strip():
            return []
        
        # Clean and tokenize text
        cleaned_text = self._clean_text(text)
        tokens = self._tokenize(cleaned_text)
        
        # Filter and score tokens
        filtered_tokens = self._filter_tokens(tokens)
        scored_tokens = self._score_tokens(filtered_tokens, text)
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(scored_tokens.items(), key=lambda x: x[1], reverse=True)
        keywords = [keyword for keyword, score in sorted_keywords[:self.max_keywords]]
        
        return keywords
    
    def extract_from_prompt(self, prompt: str) -> List[str]:
        """
        Extract keywords specifically from a Stable Diffusion prompt.
        
        This method gives higher priority to visual and descriptive terms
        that are more likely to yield good Pexels search results.
        
        Args:
            prompt: Stable Diffusion prompt text
            
        Returns:
            List of extracted keywords
        """
        keywords = self.extract_keywords(prompt)
        
        # Boost keywords that are likely to be good for image search
        visual_keywords = []
        for keyword in keywords:
            if (keyword.lower() in self.VISUAL_NOUNS or 
                keyword.lower() in self.DESCRIPTIVE_ADJECTIVES or
                self._is_visual_term(keyword)):
                visual_keywords.append(keyword)
        
        # If we have visual keywords, prioritize them
        if visual_keywords:
            return visual_keywords[:self.max_keywords]
        
        return keywords
    
    def combine_and_filter(self, text_keywords: List[str], prompt_keywords: List[str]) -> List[str]:
        """
        Combine keywords from text and prompt, removing duplicates and prioritizing.
        
        Args:
            text_keywords: Keywords extracted from original text
            prompt_keywords: Keywords extracted from SD prompt
            
        Returns:
            Combined and deduplicated list of keywords
        """
        # Create a combined list with priority to prompt keywords
        combined = []
        
        # Add prompt keywords first (higher priority)
        for keyword in prompt_keywords:
            if keyword not in combined:
                combined.append(keyword)
        
        # Add text keywords that aren't already included
        for keyword in text_keywords:
            if keyword not in combined:
                combined.append(keyword)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in combined:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        return unique_keywords[:self.max_keywords]
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing punctuation and converting to lowercase."""
        # Remove punctuation except hyphens and apostrophes within words
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens (words)."""
        return text.split()
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens to remove stop words and very short words."""
        filtered = []
        for token in tokens:
            # Skip stop words
            if token in self.STOP_WORDS:
                continue
            # Skip very short words (less than 3 characters)
            if len(token) < 3:
                continue
            # Skip pure numbers
            if token.isdigit():
                continue
            filtered.append(token)
        return filtered
    
    def _score_tokens(self, tokens: List[str], original_text: str) -> dict:
        """Score tokens based on various criteria."""
        scores = Counter(tokens)
        
        # Boost scores based on criteria
        for token in scores:
            score = scores[token]
            
            # Boost for being in visual nouns
            if token in self.VISUAL_NOUNS:
                score *= 3
            
            # Boost for being in descriptive adjectives
            if token in self.DESCRIPTIVE_ADJECTIVES:
                score *= 2
            
            # Boost for longer words (likely more specific)
            if len(token) > 6:
                score *= 1.5
            
            # Boost for words that appear in capitalized form in original text
            if self._appears_capitalized(token, original_text):
                score *= 1.2
            
            scores[token] = score
        
        return dict(scores)
    
    def _is_visual_term(self, word: str) -> bool:
        """Check if a word is likely to be visual/concrete."""
        visual_suffixes = ['tion', 'sion', 'ment', 'ness', 'ness', 'ity', 'ty']
        visual_prefixes = ['photo', 'image', 'view', 'scene', 'shot']
        
        word_lower = word.lower()
        
        # Check for visual-related prefixes/suffixes
        for prefix in visual_prefixes:
            if word_lower.startswith(prefix):
                return True
        
        # Check for common visual terms
        visual_terms = [
            'photo', 'picture', 'image', 'view', 'scene', 'shot', 'frame',
            'landscape', 'portrait', 'still', 'capture', 'moment', 'light',
            'color', 'shadow', 'reflection', 'water', 'sky', 'mountain', 'tree'
        ]
        
        for term in visual_terms:
            if term in word_lower:
                return True
        
        return False
    
    def _appears_capitalized(self, word: str, text: str) -> bool:
        """Check if a word appears capitalized in the original text."""
        # Look for the word with capital first letter in original text
        pattern = r'\b' + re.escape(word.capitalize()) + r'\b'
        return bool(re.search(pattern, text))


def extract_search_query(text: str, sd_prompt: str, max_keywords: int = 3) -> str:
    """
    Convenience function to extract search query from text and SD prompt.
    
    Args:
        text: Original paragraph text
        sd_prompt: Generated Stable Diffusion prompt
        max_keywords: Maximum number of keywords (default: 3)
        
    Returns:
        Optimized search query string
    """
    extractor = KeywordExtractor(max_keywords=max_keywords)
    
    # Extract keywords from both sources
    text_keywords = extractor.extract_keywords(text)
    prompt_keywords = extractor.extract_from_prompt(sd_prompt)
    
    # Combine and filter
    combined_keywords = extractor.combine_and_filter(text_keywords, prompt_keywords)
    
    # Return as space-separated string
    return " ".join(combined_keywords)