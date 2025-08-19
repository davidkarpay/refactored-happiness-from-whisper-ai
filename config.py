"""
Configuration settings for the Legal Transcription Toolkit
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = Path(os.getenv('TRANSCRIPTION_OUTPUT', BASE_DIR / 'output'))
TEMP_DIR = Path(os.getenv('TEMP_DIR', BASE_DIR / 'temp'))

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Model settings
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'base')
DEFAULT_DEVICE = os.getenv('DEFAULT_DEVICE', 'cpu')
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en')

# Processing settings
CHUNK_DURATION = int(os.getenv('CHUNK_DURATION_SECONDS', '120'))
OVERLAP_DURATION = int(os.getenv('OVERLAP_SECONDS', '30'))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70'))

# Speaker settings
DEFAULT_NUM_SPEAKERS = int(os.getenv('DEFAULT_NUM_SPEAKERS', '4'))
ENABLE_PYANNOTE = os.getenv('ENABLE_PYANNOTE', 'false').lower() == 'true'

# API keys (optional)
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Processing modes configuration
PROCESSING_MODES = {
    'fast': {
        'model': 'tiny',
        'chunk_duration': 300,
        'overlap': 30,
        'quality_assurance': False,
        'confidence_target': 0.80
    },
    'balanced': {
        'model': 'base',
        'chunk_duration': 300,
        'overlap': 30,
        'quality_assurance': True,
        'confidence_target': 0.85
    },
    'accuracy': {
        'model': 'large',
        'chunk_duration': 180,
        'overlap': 60,
        'quality_assurance': True,
        'confidence_target': 0.90
    },
    'high_confidence': {
        'model': 'large-v3',
        'chunk_duration': 120,
        'overlap': 30,
        'quality_assurance': True,
        'confidence_target': 0.90
    }
}