"""
Hybrid image generation step using Pexels with optional Stable Diffusion fallback.
"""

from __future__ import annotations

import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .base import PipelineStep
from .image_generator import ImageGeneratorStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class HybridImageGeneratorStep(PipelineStep[List[Dict[str, Any]], List[Path]]):
    """
    Generate images using Pexels first, with optional Stable Diffusion fallback.

    Input: List of paragraph dictionaries
    Output: List of paths to images
    """

    name = "hybrid_image_generation"
    description = "Fetch images from Pexels (LLM keywords), fallback to Stable Diffusion"

    _llm_cache: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}
    _spacy_nlp = None

    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)
        self._sd_generator = ImageGeneratorStep(config, cache)

    def run(self, paragraphs: List[Dict[str, Any]]) -> List[Path]:
        image_paths: List[Path] = []
        for idx, para in enumerate(paragraphs):
            cached_path = self.cache.get_cached_image(
                para["text"], idx, self.config.images_dir
            )
            if cached_path is not None:
                image_paths.append(cached_path)
                continue

            fetched = self._fetch_from_pexels(para["text"], idx, total=len(paragraphs))
            if fetched is not None:
                image_paths.append(fetched)
                continue

            if self.config.pexels_fallback_to_sd:
                img_path = self._sd_generator.generate_for_paragraph(
                    para, idx, total=len(paragraphs)
                )
                image_paths.append(img_path)
                continue

            raise RuntimeError(
                "Pexels failed to return a usable image and Stable Diffusion fallback "
                "is disabled. Use --sd-fallback to enable fallback."
            )

        print(f"Generated {len(image_paths)} images (Pexels + optional SD)")
        return image_paths

    def _fetch_from_pexels(self, text: str, idx: int, total: int) -> Optional[Path]:
        api_key = self.config.pexels_api_key or os.environ.get("PEXELS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PEXELS_API_KEY is not set. Add it to the environment or use setup_env.py."
            )

        keywords = self._extract_keywords(text)
        img_path = self._try_pexels_queries(api_key, keywords, text, idx, total)
        if img_path is not None:
            return img_path

        # If no relevant results, retry with abbreviated prompt (top keywords only)
        if len(keywords) > 2:
            short_keywords = keywords[:2]
            print(f"No relevant Pexels match; retrying with abbreviated prompt: {short_keywords}")
            img_path = self._try_pexels_queries(api_key, short_keywords, text, idx, total)
            if img_path is not None:
                return img_path

        return None

    def _try_pexels_queries(
        self,
        api_key: str,
        keywords: List[str],
        paragraph_text: str,
        idx: int,
        total: int,
    ) -> Optional[Path]:
        queries = self._build_queries(keywords)
        for query in queries:
            print(f"Pexels search query: {query}")
            photos = self._search_pexels(api_key, query)
            if not photos:
                continue

            best = self._select_best_photo(photos, keywords)
            if best is None:
                continue

            img_path = self._download_photo(best, idx, total=total)
            if img_path is not None:
                self.cache.save_image_cache(paragraph_text, idx, img_path)
                return img_path
        return None

    def _extract_keywords(self, text: str) -> List[str]:
        if self.config.llm_keyword_extractor:
            try:
                return self._extract_keywords_spacy(text)
            except Exception as e:
                print(f"spaCy keyword extraction failed, falling back: {e}")
        return self._extract_keywords_simple(text)

    def _extract_keywords_spacy(self, text: str) -> List[str]:
        nlp = self._get_spacy()
        doc = nlp(text)

        # Collect noun chunks and important named entities
        phrases: List[str] = []
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if 1 <= len(phrase.split()) <= 4:
                phrases.append(phrase)
        for ent in doc.ents:
            phrase = ent.text.strip().lower()
            if 1 <= len(phrase.split()) <= 4:
                phrases.append(phrase)

        # Score phrases by frequency and position
        freq: Dict[str, int] = {}
        for p in phrases:
            freq[p] = freq.get(p, 0) + 1

        ranked = sorted(
            freq.items(),
            key=lambda kv: (-kv[1], len(kv[0].split()), kv[0])
        )

        keywords = [p for p, _ in ranked]
        keywords = self._normalize_keywords(keywords)

        min_k = max(1, self.config.llm_min_keywords)
        max_k = max(min_k, self.config.llm_max_keywords)
        if len(keywords) < min_k:
            keywords = self._pad_keywords(keywords, text)
        return keywords[:max_k]

    def _get_spacy(self):
        if self._spacy_nlp is not None:
            return self._spacy_nlp
        try:
            import spacy
        except Exception as e:
            raise RuntimeError("spaCy is not installed. Add spacy to requirements.") from e
        try:
            self._spacy_nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from e
        return self._spacy_nlp

    def _extract_keywords_llm(self, text: str) -> List[str]:
        tokenizer, model = self._get_llm()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        min_k = max(1, self.config.llm_min_keywords)
        max_k = max(min_k, self.config.llm_max_keywords)
        prompt = (
            "You are extracting search keywords for image retrieval. "
            f"Return {min_k}-{max_k} concrete visual keywords or short phrases "
            "(1-4 words each). Avoid verbs and abstract concepts. "
            "Respond as a JSON array of strings only.\n\n"
            f"Text: {text}\nKeywords:"
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        extracted = decoded.split("Keywords:")[-1].strip()

        # Try JSON parsing first
        try:
            parsed = json.loads(extracted)
            keywords = [str(k).strip() for k in parsed if str(k).strip()]
        except json.JSONDecodeError:
            # Fallback: try bracketed list or split on commas
            match = re.search(r"\[(.*?)\]", extracted, re.DOTALL)
            if match:
                inner = match.group(1)
                keywords = [k.strip().strip("\"'") for k in inner.split(",") if k.strip()]
            else:
                keywords = [k.strip() for k in extracted.split(",") if k.strip()]

        keywords = self._normalize_keywords(keywords)
        if len(keywords) < min_k:
            keywords = self._pad_keywords(keywords, text)
        return keywords[:max_k]

    def _get_llm(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        model_name = self.config.llm_model_name
        if model_name in self._llm_cache:
            return self._llm_cache[model_name]

        print(f"Loading LLM for keywords: {model_name} (CPU)")
        if self.config.llm_quantization:
            print("LLM quantization requested, but CPU quantization is not enabled; using fp32.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        # Create a safe generation config that does not rely on model.config.pad_token_id
        model.generation_config = GenerationConfig(
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        model.to("cpu")
        model.eval()

        self._llm_cache[model_name] = (tokenizer, model)
        return tokenizer, model


    def _extract_keywords_simple(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        stopwords = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as",
            "is", "are", "was", "were", "be", "been", "by", "at", "from", "that",
            "this", "it", "its", "their", "his", "her", "they", "he", "she", "we",
            "you", "i", "my", "your", "our", "but", "not", "so", "if", "then"
        }
        words = [t for t in tokens if t not in stopwords and len(t) > 2]
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        keywords = [w for w, _ in ranked[:3]]
        if len(keywords) < 3:
            keywords = self._pad_keywords(keywords, text)
        return keywords[:3]

    def _pad_keywords(self, keywords: List[str], text: str) -> List[str]:
        if not keywords:
            return ["scene", "people", "landscape"]
        if len(keywords) == 1:
            return keywords + ["scene", "landscape"]
        if len(keywords) == 2:
            return keywords + ["scene"]
        return keywords

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as",
            "is", "are", "was", "were", "be", "been", "by", "at", "from", "that",
            "this", "it", "its", "their", "his", "her", "they", "he", "she", "we",
            "you", "i", "my", "your", "our", "but", "not", "so", "if", "then",
            "there", "here", "into", "out", "up", "down", "over", "under", "again",
            "about", "after", "before", "during", "while", "because", "which", "who",
            "whom", "what", "when", "where", "why", "how", "also", "just", "very",
        }
        normalized = []
        for k in keywords:
            cleaned = re.sub(r"[^A-Za-z0-9' -]", "", k).strip()
            if not cleaned:
                continue
            if cleaned.lower() in stopwords:
                continue
            if all(part.lower() in stopwords for part in cleaned.split()):
                continue
            if cleaned:
                normalized.append(cleaned)
        return normalized

    def _build_queries(self, keywords: List[str]) -> List[str]:
        if not keywords:
            return []
        base = " ".join(keywords)
        queries = [base]
        if len(keywords) >= 2:
            queries.append(" ".join(keywords[:2]))
        queries.append(keywords[0])
        return queries

    def _search_pexels(self, api_key: str, query: str) -> List[Dict[str, Any]]:
        params = {
            "query": query,
            "per_page": self.config.pexels_per_page,
            "orientation": self.config.pexels_orientation,
        }
        headers = {"Authorization": api_key}

        try:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                params=params,
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                print(f"Pexels search failed ({resp.status_code}): {resp.text}")
                return []
            data = resp.json()
            return data.get("photos", [])
        except requests.RequestException as e:
            print(f"Pexels request failed: {e}")
            return []

    def _select_best_photo(
        self, photos: List[Dict[str, Any]], keywords: List[str]
    ) -> Optional[Dict[str, Any]]:
        if not photos:
            return None

        def score(p: Dict[str, Any]) -> int:
            width = int(p.get("width", 0))
            height = int(p.get("height", 0))
            is_landscape = 1 if width >= height else 0
            meets_min = 1 if (
                width >= self.config.pexels_min_width and
                height >= self.config.pexels_min_height
            ) else 0
            relevance = self._keyword_match_score(p, keywords)
            return (
                (relevance * 100_000_000)
                + (meets_min * 10_000_000)
                + (is_landscape * 1_000_000)
                + (width * height)
            )

        best = max(photos, key=score, default=None)
        if best is None:
            return None
        if self._keyword_match_score(best, keywords) <= 0:
            return None
        return best

    def _keyword_match_score(self, photo: Dict[str, Any], keywords: List[str]) -> int:
        alt = (photo.get("alt") or "").lower()
        url = (photo.get("url") or "").lower()
        text = f"{alt} {url}"

        matches = 0
        for kw in keywords:
            kw_clean = kw.lower().strip()
            if not kw_clean:
                continue
            # Phrase match: all words in keyword phrase must appear
            parts = [p for p in kw_clean.split() if p]
            if parts and all(p in text for p in parts):
                matches += 1
        return matches

    def _download_photo(self, photo: Dict[str, Any], idx: int, total: int) -> Optional[Path]:
        src = photo.get("src", {})
        url = src.get("large2x") or src.get("large") or src.get("original")
        if not url:
            return None

        try:
            print(f"Fetching Pexels image {idx+1}/{total}...")
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                print(f"Pexels download failed ({resp.status_code})")
                return None
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img_path = self.config.images_dir / f"{idx:03d}.png"
            img.save(img_path)
            return img_path
        except Exception as e:
            print(f"Failed to download/convert Pexels image: {e}")
            return None
