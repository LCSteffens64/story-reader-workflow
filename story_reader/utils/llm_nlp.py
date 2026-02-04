"""
LLM-based keyword extraction for Pexels search queries.
Uses small, efficient models from HuggingFace for intelligent keyword extraction.
"""

import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LLMKeywordExtractor:
    """
    Extracts keywords using a small LLM for more intelligent and specific results.
    
    Uses models like Phi-2 or TinyLlama that are under 2GB with quantization.
    """
    
    def __init__(self, model_name: str = "microsoft/phi-2", device: str = "cpu", quantize: bool = True):
        """
        Initialize the LLM keyword extractor.
        
        Args:
            model_name: HuggingFace model name (default: microsoft/phi-2)
            device: Device to run model on (cpu/cuda)
            quantize: Whether to use 4-bit quantization for memory efficiency
        """
        self.model_name = model_name
        self.device = device
        self.quantize = quantize
        
        # Initialize model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
    
    def _load_tokenizer(self):
        """Load the tokenizer for the specified model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer for {self.model_name}: {e}")
    
    def _load_model(self):
        """Load the model with optional quantization."""
        try:
            if self.quantize:
                # Use 4-bit quantization for memory efficiency
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
            
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_name}: {e}")
    
    def extract_keywords(self, text: str, sd_prompt: str, max_keywords: int = 3) -> List[str]:
        """
        Extract keywords using the LLM.
        
        Args:
            text: Original paragraph text
            sd_prompt: Generated Stable Diffusion prompt
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        if not text or not text.strip():
            return []
        
        # Build the prompt for keyword extraction
        prompt = self._build_extraction_prompt(text, sd_prompt, max_keywords)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(inputs, max_new_tokens=50, do_sample=False, temperature=0.1, pad_token_id=self.tokenizer.eos_token_id)
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract keywords from response
            keywords = self._parse_keywords(response, max_keywords)
            return keywords            
            
        except Exception as e:
            print(f"LLM keyword extraction failed: {e}")
            # Fallback to simple extraction
            return self._simple_keyword_extraction(text, max_keywords)
    
    def _build_extraction_prompt(self, text: str, sd_prompt: str, max_keywords: int) -> str:
        """Build the prompt for the LLM."""
        return f"""Extract {max_keywords} specific keywords for Pexels image search from this text.
Focus on concrete visual elements, objects, settings, and actions. Avoid abstract concepts, emotions, or generic adjectives.

Guidelines:
- Extract concrete nouns and visual elements
- Include specific objects, locations, and actions
- Use multi-word phrases when they describe a single concept
- Prioritize elements that would appear in a photograph
- Avoid: beautiful, amazing, good, bad, happy, sad, feeling, emotion, abstract, concept

Text: "{text}"
Stable Diffusion Prompt: "{sd_prompt}"

Return keywords as a comma-separated list.
Examples:

Input: "A cat sleeping on a windowsill in the afternoon sun"
Output: cat, windowsill, sleeping, afternoon sun, indoor

Input: "Children playing in a sunlit meadow with flowers"
Output: children, meadow, playing, sunlight, flowers, grass

Input: "The old lighthouse stood proudly against the stormy sea"
Output: lighthouse, stormy sea, coastal scene, dramatic weather, ocean waves

Now extract keywords from:
Input: "{text}"
Output:"""
    
    def _parse_keywords(self, response: str, max_keywords: int) -> List[str]:
        """Parse keywords from LLM response."""
        # Extract the part after "Output:" if present
        if "Output:" in response:
            response = response.split("Output:")[-1]
        
        # Split by comma and clean up
        keywords = [kw.strip().lower() for kw in response.split(',') if kw.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen and len(keyword) > 2:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:max_keywords]
    
    def _simple_keyword_extraction(self, text: str, max_keywords: int) -> List[str]:
        """Simple fallback keyword extraction."""
        # Remove punctuation and split into words
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common words
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'have', 'has', 'had', 'was', 'were', 'be',
            'beautiful', 'amazing', 'stunning', 'gorgeous', 'nice', 'good', 'bad', 'old',
            'new', 'big', 'small', 'large', 'little', 'many', 'much', 'some', 'any'
        }
        
        filtered_words = [word for word in words if word not in common_words]
        
        # Return most frequent words
        from collections import Counter
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(max_keywords)]


def extract_search_query(text: str, sd_prompt: str, model_name: str = "microsoft/phi-2", 
                        device: str = "cpu", max_keywords: int = 3) -> str:
    """
    Convenience function to extract search query using LLM.
    
    Args:
        text: Original paragraph text
        sd_prompt: Generated Stable Diffusion prompt
        model_name: HuggingFace model name
        device: Device to run model on
        max_keywords: Maximum number of keywords
        
    Returns:
        Optimized search query string
    """
    extractor = LLMKeywordExtractor(model_name=model_name, device=device)
    keywords = extractor.extract_keywords(text, sd_prompt, max_keywords)
    return " ".join(keywords)