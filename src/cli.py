#!/usr/bin/env python3
"""
Legal Transcription CLI Tool
A command-line interface for transcribing legal depositions with speaker identification
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import our modules
from .audio_transcriber import AudioTranscriber
from .deposition_transcriber import DepositionTranscriber
from .high_confidence_transcriber import HighConfidenceDepositionTranscriber
import config

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Legal Transcription Toolkit - Transcribe legal depositions with speaker identification"""
    pass

@cli.command('record')
@click.option('--model', '-m', default=config.DEFAULT_MODEL, 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model size')
@click.option('--device', '-d', default=config.DEFAULT_DEVICE,
              type=click.Choice(['cpu', 'cuda']),
              help='Device to use for inference')
@click.option('--language', '-l', default=config.DEFAULT_LANGUAGE,
              help='Language code (e.g., en, es, fr)')
def record(model, device, language):
    """Start recording and transcribing audio in real-time"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]Real-time Audio Transcription[/bold cyan]\\n"
        f"Model: {model} | Device: {device} | Language: {language}",
        border_style="cyan"
    ))
    
    transcriber = AudioTranscriber(model_size=model, device=device, language=language)
    
    try:
        transcriber.load_model()
        transcriber.start_recording()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    finally:
        transcriber.cleanup()

@cli.command('transcribe-deposition')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--mode', default='balanced',
              type=click.Choice(['fast', 'balanced', 'accuracy']),
              help='Processing mode: fast (~10min), balanced (~20min), accuracy (~45min)')
@click.option('--model', '-m', default=None,
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Override model size (overrides mode selection)')
@click.option('--device', '-d', default=config.DEFAULT_DEVICE,
              type=click.Choice(['cpu', 'cuda']),
              help='Device to use for inference')
@click.option('--language', '-l', default=config.DEFAULT_LANGUAGE,
              help='Language code (e.g., en, es, fr)')
@click.option('--speakers', '-s', type=int, default=config.DEFAULT_NUM_SPEAKERS,
              help='Expected number of speakers')
@click.option('--case-number', '-c', default=None,
              help='Case number for the deposition (optional)')
@click.option('--resume', is_flag=True,
              help='Resume from previous interrupted transcription')
def transcribe_deposition(input_file, mode, model, device, language, speakers, case_number, resume):
    """Transcribe a deposition with speaker identification and quality assurance"""
    console = Console()
    
    # Determine model based on mode if not explicitly specified
    if not model:
        model = config.PROCESSING_MODES[mode]['model']
    
    console.print(Panel.fit(
        "[bold cyan]Deposition Transcription[/bold cyan]\\n"
        f"File: {Path(input_file).name}\\n"
        f"Mode: {mode} | Model: {model} | Device: {device} | Language: {language}\\n"
        f"Expected Speakers: {speakers or 'Auto-detect'}",
        border_style="cyan"
    ))
    
    transcriber = DepositionTranscriber(
        model_size=model, 
        device=device, 
        language=language,
        num_speakers=speakers,
        processing_mode=mode
    )
    
    try:
        segments, confidence = transcriber.transcribe_deposition(input_file, case_number, resume=resume)
        
        # Display speaker summary
        speakers_found = set(s.get('role', 'Unknown') for s in segments)
        console.print(f"\\n[green]Transcription completed with {confidence:.1%} confidence[/green]")
        console.print(f"[green]Identified speakers: {', '.join(speakers_found)}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during deposition transcription: {e}[/red]")
        sys.exit(1)

@cli.command('transcribe-hq')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--speakers', '-s', type=int, default=config.DEFAULT_NUM_SPEAKERS,
              help='Expected number of speakers')
@click.option('--case-number', '-c', default=None,
              help='Case number for the deposition (optional)')
@click.option('--device', '-d', default=config.DEFAULT_DEVICE,
              type=click.Choice(['cpu', 'cuda']),
              help='Device to use for inference')
def transcribe_hq(input_file, speakers, case_number, device):
    """High-confidence deposition transcription with 90%+ accuracy target"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]High-Confidence Deposition Transcription[/bold cyan]\\n"
        f"File: {Path(input_file).name}\\n"
        f"Target Confidence: 90%+ | Device: {device}\\n"
        f"Processing: Two-pass with validation",
        border_style="cyan"
    ))
    
    transcriber = HighConfidenceDepositionTranscriber(
        model_size="large-v3",
        device=device,
        language="en"
    )
    
    try:
        segments, confidence = transcriber.transcribe_high_confidence(
            input_file, 
            num_speakers=speakers,
            case_number=case_number
        )
        
        # Display results
        if confidence >= 0.90:
            status_color = "green"
            status = "CERTIFIED"
        elif confidence >= 0.80:
            status_color = "yellow"
            status = "REVIEW RECOMMENDED"
        else:
            status_color = "red"
            status = "MANUAL REVIEW REQUIRED"
        
        console.print(f"\\n[{status_color}]Transcription Status: {status}[/{status_color}]")
        console.print(f"[{status_color}]Overall Confidence: {confidence:.1%}[/{status_color}]")
        
        # Show speaker summary
        speakers_found = set(s.get('speaker', 'Unknown') for s in segments)
        console.print(f"[green]Identified {len(speakers_found)} speakers[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during high-confidence transcription: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@cli.command('list-sessions')
def list_sessions():
    """List all transcription sessions"""
    console = Console()
    
    # Check different output directories
    directories = [
        (config.OUTPUT_DIR / "transcriptions", "Real-time Sessions"),
        (config.OUTPUT_DIR / "depositions", "Deposition Sessions"), 
        (config.OUTPUT_DIR / "depositions_hq", "High-Confidence Sessions")
    ]
    
    for directory, title in directories:
        if directory.exists() and any(directory.iterdir()):
            console.print(f"\\n[bold cyan]{title}[/bold cyan]")
            
            table = Table()
            table.add_column("Session ID", style="cyan")
            table.add_column("Date/Time", style="white")
            table.add_column("Files", style="green")
            
            for session_dir in sorted(directory.iterdir(), reverse=True):
                if session_dir.is_dir():
                    session_id = session_dir.name
                    # Parse datetime from session ID
                    try:
                        dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
                        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        date_str = "Unknown"
                    
                    # Check which files exist
                    files = []
                    for file in session_dir.glob("*.txt"):
                        if "transcript" in file.name:
                            files.append("transcript")
                    for file in session_dir.glob("*.csv"):
                        files.append("csv")
                    for file in session_dir.glob("*.json"):
                        files.append("json")
                    
                    if files:  # Only show sessions with output files
                        table.add_row(session_id, date_str, ", ".join(files))
            
            if table.rows:
                console.print(table)
            else:
                console.print("[yellow]No sessions found[/yellow]")

if __name__ == "__main__":
    cli()