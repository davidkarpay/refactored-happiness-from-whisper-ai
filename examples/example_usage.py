#!/usr/bin/env python3
"""
Example usage of the Legal Transcription Toolkit
This file demonstrates how to use the transcription classes programmatically
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deposition_transcriber import DepositionTranscriber
from high_confidence_transcriber import HighConfidenceDepositionTranscriber
import config

def example_basic_transcription():
    """Example: Basic deposition transcription"""
    print("=" * 60)
    print("EXAMPLE: Basic Deposition Transcription")
    print("=" * 60)
    
    # Initialize transcriber
    transcriber = DepositionTranscriber(
        model_size="base",
        device="cpu",
        language="en",
        num_speakers=4,
        processing_mode="balanced"
    )
    
    # Example audio file path (replace with your actual file)
    audio_file = "path/to/your/deposition.mp3"
    
    # Check if file exists (for demo purposes)
    if not Path(audio_file).exists():
        print(f"Demo: Would transcribe {audio_file}")
        print("Features:")
        print("- Speaker identification (4 speakers)")
        print("- Legal formatting")
        print("- Confidence scoring")
        print("- Multiple output formats (TXT, CSV, JSON)")
        return
    
    try:
        # Perform transcription
        segments, confidence = transcriber.transcribe_deposition(
            input_file=audio_file,
            case_number="EXAMPLE-2024-001"
        )
        
        print(f"Transcription completed!")
        print(f"Overall confidence: {confidence:.1%}")
        print(f"Number of segments: {len(segments)}")
        
        # Display first few segments
        print("\\nFirst 3 segments:")
        for i, segment in enumerate(segments[:3]):
            print(f"{i+1}. [{segment['start']:.1f}s] {segment.get('role', 'Unknown')}: {segment['text'][:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")

def example_high_confidence_transcription():
    """Example: High-confidence transcription with validation"""
    print("\\n" + "=" * 60)
    print("EXAMPLE: High-Confidence Transcription")
    print("=" * 60)
    
    # Initialize high-confidence transcriber
    transcriber = HighConfidenceDepositionTranscriber(
        model_size="large-v3",
        device="cpu",
        language="en"
    )
    
    audio_file = "path/to/your/deposition.mp3"
    
    if not Path(audio_file).exists():
        print(f"Demo: Would perform high-confidence transcription of {audio_file}")
        print("Features:")
        print("- Two-pass processing")
        print("- 90%+ accuracy target")
        print("- Validation report")
        print("- Court-admissible quality")
        print("- [UNCLEAR] markers for review")
        return
    
    try:
        # Perform high-confidence transcription
        segments, confidence = transcriber.transcribe_high_confidence(
            input_file=audio_file,
            num_speakers=4,
            case_number="EXAMPLE-2024-001"
        )
        
        print(f"High-confidence transcription completed!")
        print(f"Overall confidence: {confidence:.1%}")
        
        if confidence >= 0.90:
            print("✅ CERTIFIED - Court admissible quality")
        elif confidence >= 0.80:
            print("⚠️  REVIEW RECOMMENDED - Good quality, minor review needed")
        else:
            print("❌ MANUAL REVIEW REQUIRED - Significant review needed")
        
        # Count segments by confidence
        high_conf = sum(1 for s in segments if s.get('confidence', 0) >= 0.85)
        low_conf = sum(1 for s in segments if s.get('confidence', 0) < 0.70)
        
        print(f"High confidence segments: {high_conf}/{len(segments)}")
        print(f"Low confidence segments: {low_conf}/{len(segments)}")
        
    except Exception as e:
        print(f"Error: {e}")

def example_configuration():
    """Example: Working with configuration"""
    print("\\n" + "=" * 60)
    print("EXAMPLE: Configuration Options")
    print("=" * 60)
    
    print("Current Configuration:")
    print(f"- Output Directory: {config.OUTPUT_DIR}")
    print(f"- Default Model: {config.DEFAULT_MODEL}")
    print(f"- Default Device: {config.DEFAULT_DEVICE}")
    print(f"- Default Language: {config.DEFAULT_LANGUAGE}")
    print(f"- Min Confidence: {config.MIN_CONFIDENCE_THRESHOLD}")
    
    print("\\nProcessing Modes Available:")
    for mode, settings in config.PROCESSING_MODES.items():
        print(f"- {mode.upper()}: {settings['model']} model, {settings['confidence_target']:.0%} target")
    
    print("\\nEnvironment Variables (set these in .env):")
    print("- HUGGINGFACE_TOKEN: For advanced speaker diarization")
    print("- TRANSCRIPTION_OUTPUT: Custom output directory")
    print("- DEFAULT_MODEL: Override default model")

def example_quality_metrics():
    """Example: Understanding quality metrics"""
    print("\\n" + "=" * 60)
    print("EXAMPLE: Quality Metrics Explained")
    print("=" * 60)
    
    print("Confidence Levels:")
    print("- 90-100%: Excellent (court admissible)")
    print("- 80-90%:  Good (minor review recommended)")
    print("- 70-80%:  Fair (review recommended)")  
    print("- <70%:    Poor (manual review required)")
    
    print("\\nOutput Files:")
    print("1. transcript_[timestamp].txt - Formatted legal transcript")
    print("2. details_[timestamp].csv - Segment-by-segment data")
    print("3. data_[timestamp].json - Complete transcription data")
    print("4. validation_report_[timestamp].txt - Quality analysis")
    
    print("\\nSpeaker Roles Automatically Assigned:")
    for role in DepositionTranscriber.SPEAKER_ROLES.values():
        print(f"- {role}")

if __name__ == "__main__":
    print("Legal Transcription Toolkit - Example Usage")
    print("=" * 80)
    
    # Run examples
    example_configuration()
    example_quality_metrics()
    example_basic_transcription()
    example_high_confidence_transcription()
    
    print("\\n" + "=" * 80)
    print("For more information, see README.md or run:")
    print("python -m src.cli --help")
    print("=" * 80)