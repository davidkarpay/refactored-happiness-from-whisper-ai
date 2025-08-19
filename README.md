# Legal Transcription Toolkit

A powerful, open-source transcription system specifically designed for legal depositions using OpenAI Whisper with advanced speaker identification and quality validation.

## 🎯 Features

### Core Capabilities
- **Real-time Audio Transcription** - Live recording and transcription
- **Legal Deposition Transcription** - Specialized for legal proceedings
- **High-Confidence Mode** - 90%+ accuracy with validation reports
- **Speaker Diarization** - Automatic speaker identification and role assignment
- **Quality Assurance** - Multi-pass processing with confidence scoring

### Quality Levels
- **Fast Mode** (`~10 minutes`): Tiny model, 80%+ confidence
- **Balanced Mode** (`~20 minutes`): Base model, 85%+ confidence  
- **Accuracy Mode** (`~45 minutes`): Large model, 90%+ confidence
- **High-Confidence Mode** (`~60 minutes`): Large-v3 model with validation, 90%+ target

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/davidkarpay/refactored-happiness-from-whisper-ai.git
cd refactored-happiness-from-whisper-ai

# Install dependencies
pip install -r requirements.txt

# Optional: Install high-quality dependencies
pip install -r requirements-dev.txt
```

### Basic Usage

```bash
# Record and transcribe in real-time
python -m src.cli record --model base

# Transcribe a deposition (balanced mode)
python -m src.cli transcribe-deposition path/to/audio.mp3 --speakers 4

# High-confidence transcription with validation
python -m src.cli transcribe-hq path/to/audio.mp3 --speakers 4

# List all transcription sessions
python -m src.cli list-sessions
```

## 📋 Requirements

### Core Requirements
- Python 3.8+
- OpenAI Whisper
- PyTorch
- Rich (for CLI interface)
- Click (for command-line parsing)

### Optional Dependencies
- `pyaudio` - For real-time recording
- `pyannote.audio` - For advanced speaker diarization
- `webrtcvad` - For voice activity detection
- `scipy` - For audio signal processing

### Hardware Recommendations
- **CPU**: Multi-core processor (transcription is CPU-intensive)
- **Memory**: 8GB+ RAM (16GB+ recommended for large files)
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)

## 🎛️ Configuration

### Environment Variables

Copy `.env.template` to `.env` and configure:

```bash
# API Keys (optional)
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here

# Paths
TRANSCRIPTION_OUTPUT=./output
TEMP_DIR=./temp

# Processing Settings
DEFAULT_MODEL=base
DEFAULT_DEVICE=cpu
MIN_CONFIDENCE_THRESHOLD=0.70
```

### Processing Modes

| Mode | Model | Processing Time | Target Confidence | Best For |
|------|-------|----------------|------------------|----------|
| **Fast** | tiny | ~10 minutes | 80%+ | Quick drafts, real-time |
| **Balanced** | base | ~20 minutes | 85%+ | General use, good quality |
| **Accuracy** | large | ~45 minutes | 90%+ | Important documents |
| **High-Confidence** | large-v3 | ~60 minutes | 90%+ | Court admissible |

## 📁 Output Formats

### Legal Transcript Format
```
================================================================================
DEPOSITION TRANSCRIPT
================================================================================
Case: [Case Number]
Overall Confidence: XX.X%
Quality Status: CERTIFIED/REVIEW NEEDED
================================================================================

Court Reporter:
    [Transcript content with proper legal formatting]

Defense Counsel:
    [Transcript content]
...
```

### Additional Outputs
- **CSV**: Detailed segment data with timestamps and confidence scores
- **JSON**: Complete transcription data for programmatic access
- **Validation Report**: Quality metrics and review recommendations

## 🔧 Advanced Usage

### Speaker Role Assignment
The system automatically identifies and assigns roles:
- Court Reporter
- Defense Counsel  
- State/Plaintiff
- Witness
- Judge
- Other Party

### Quality Validation
High-confidence mode includes:
- **Two-pass transcription** for low-confidence segments
- **Confidence scoring** at word and segment levels
- **Validation reports** with quality metrics
- **[UNCLEAR]** markers for segments requiring review

### Example Commands

```bash
# Fast transcription for quick review
python -m src.cli transcribe-deposition audio.mp3 --mode fast

# High-confidence with specific case number
python -m src.cli transcribe-hq deposition.mp3 \\
    --speakers 4 \\
    --case-number "24CF008025"

# CPU vs GPU processing
python -m src.cli transcribe-hq audio.mp3 --device cuda  # GPU
python -m src.cli transcribe-hq audio.mp3 --device cpu   # CPU
```

## 🔐 Security & Privacy

### Data Protection
- **No cloud processing** - All transcription happens locally
- **Automatic .gitignore** - Sensitive files never committed
- **Configurable output paths** - Control where files are saved
- **No telemetry** - Your data stays on your machine

### Legal Compliance
- Designed for legal professional use
- Court-admissible quality with validation
- Proper timestamps and confidence scoring
- Professional transcript formatting

## 📊 Quality Metrics

### Confidence Scoring
- **Word-level confidence** - Individual word accuracy
- **Segment confidence** - Phrase/sentence accuracy  
- **Overall confidence** - Complete transcript accuracy
- **Speaker confidence** - Speaker identification accuracy

### Validation Criteria
- ≥90% overall confidence for certification
- ≤10% low-confidence segments
- Proper speaker role assignment
- Legal terminology accuracy

## 🛠️ Development

### Project Structure
```
src/
├── cli.py                          # Command-line interface
├── audio_transcriber.py            # Real-time transcription
├── deposition_transcriber.py       # Legal deposition processing
└── high_confidence_transcriber.py  # High-accuracy processing

tests/                              # Unit tests
docs/                              # Documentation
examples/                          # Usage examples
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## ⚖️ Legal Disclaimer

This tool is designed to assist legal professionals but does not replace human review. Always verify transcripts for accuracy before use in legal proceedings.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/davidkarpay/refactored-happiness-from-whisper-ai/issues)
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory

## 🏷️ Version

Current Version: 1.0.0

---

**Built with ❤️ for the legal community**