"""
Legal Transcription Toolkit
A powerful transcription system for legal depositions using OpenAI Whisper
"""

__version__ = "1.0.0"
__author__ = "Legal Transcription Team"

from .audio_transcriber import AudioTranscriber
from .deposition_transcriber import DepositionTranscriber  
from .high_confidence_transcriber import HighConfidenceDepositionTranscriber

__all__ = [
    'AudioTranscriber',
    'DepositionTranscriber', 
    'HighConfidenceDepositionTranscriber'
]