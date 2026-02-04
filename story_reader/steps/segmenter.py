"""
Paragraph segmentation step - groups transcript segments into paragraphs.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class SegmenterStep(PipelineStep[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """
    Segments transcribed text into logical paragraphs.
    
    Input: List of transcript segments from Whisper
    Output: List of paragraph dictionaries with merged text and timestamps
    
    Paragraphs are split based on:
    - Silence gaps (configurable via min_silence)
    - Maximum duration (configurable via max_paragraph_duration)
    - Maximum sentences (configurable via max_sentences)
    """
    
    name = "paragraph_segmentation"
    description = "Group transcript segments into paragraphs"
    
    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)
    
    def run(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge transcript segments into paragraphs.
        
        Args:
            segments: List of Whisper segments with text, start, end times
            
        Returns:
            List of paragraph dictionaries containing:
            - text: Combined paragraph text
            - start: Start time in seconds
            - end: End time in seconds
            - duration: Paragraph duration in seconds
        """
        paragraphs = []
        current_segments = []
        paragraph_start = None
        
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_text = seg["text"].strip()
            
            if paragraph_start is None:
                paragraph_start = seg_start
            
            # Calculate gap from previous segment
            prev_end = current_segments[-1]["end"] if current_segments else seg_start
            gap = seg_start - prev_end
            
            # Calculate current paragraph metrics
            paragraph_duration = seg_end - paragraph_start
            num_sentences = len(current_segments) + 1
            
            # Check if we should start a new paragraph
            should_split = (
                gap >= self.config.min_silence or
                paragraph_duration >= self.config.max_paragraph_duration or
                num_sentences > self.config.max_sentences
            )
            
            if should_split and current_segments:
                # Save current paragraph
                para_text = " ".join([s["text"] for s in current_segments])
                paragraphs.append({
                    "text": para_text.strip(),
                    "start": paragraph_start,
                    "end": current_segments[-1]["end"],
                    "duration": current_segments[-1]["end"] - paragraph_start,
                })
                # Start new paragraph
                current_segments = []
                paragraph_start = seg_start
            
            current_segments.append(seg)
        
        # Don't forget the last paragraph
        if current_segments:
            para_text = " ".join([s["text"] for s in current_segments])
            paragraphs.append({
                "text": para_text.strip(),
                "start": paragraph_start,
                "end": current_segments[-1]["end"],
                "duration": current_segments[-1]["end"] - paragraph_start,
            })
        
        # Save paragraphs to file for debugging/inspection
        self._save_paragraphs(paragraphs)
        
        print(f"Created {len(paragraphs)} paragraphs from {len(segments)} segments")
        return paragraphs
    
    def _save_paragraphs(self, paragraphs: List[Dict[str, Any]]) -> None:
        """Save paragraphs to JSON file for inspection."""
        output_file = self.config.paragraphs_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(paragraphs, indent=2))
