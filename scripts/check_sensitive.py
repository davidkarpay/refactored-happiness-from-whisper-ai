#!/usr/bin/env python3
"""
Security check script to prevent sensitive information from being committed
"""

import re
import sys
import os
from pathlib import Path

# Sensitive patterns to check for
SENSITIVE_PATTERNS = [
    # Case-specific information
    (r'24CF\d+', "Case number pattern"),
    (r'\bDUFFIE\b', "Defendant name"),
    (r'\bDuffy\b', "Defendant name (alt)"),
    (r'\bHollis\b', "Witness name"),
    (r'\bMartin\b', "Attorney name"),
    
    # File paths that might contain sensitive info
    (r'F:\\\\.*', "Windows file path"),
    (r'/Users/.*', "macOS user path"),
    
    # Audio file references
    (r'\.mp3\b(?!.*example)', "MP3 file reference"),
    (r'\.wav\b(?!.*temp|.*example)', "WAV file reference"),
    (r'\.m4a\b', "M4A file reference"),
    
    # Transcription content indicators
    (r'deposition.*transcript', "Transcript reference"),
    (r'sworn testimony', "Legal content"),
    
    # API keys and tokens
    (r'sk-[a-zA-Z0-9]{32,}', "OpenAI API key"),
    (r'hf_[a-zA-Z0-9]{32,}', "HuggingFace token"),
    
    # Email addresses (might contain sensitive info)
    (r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', "Email address"),
]

# Files to always ignore
IGNORE_FILES = {
    '.gitignore',
    'check_sensitive.py',
    'README.md',
    'LICENSE',
    '.env.template'
}

# Extensions to check
CHECK_EXTENSIONS = {'.py', '.txt', '.md', '.json', '.csv', '.yml', '.yaml'}

def check_file(file_path: Path) -> list:
    """Check a single file for sensitive patterns"""
    issues = []
    
    # Skip if file should be ignored
    if file_path.name in IGNORE_FILES:
        return issues
    
    # Only check certain file types
    if file_path.suffix not in CHECK_EXTENSIONS:
        return issues
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        for i, line in enumerate(content.split('\\n'), 1):
            for pattern, description in SENSITIVE_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        'file': str(file_path),
                        'line': i,
                        'pattern': description,
                        'content': line.strip()[:100] + '...' if len(line.strip()) > 100 else line.strip()
                    })
    
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
    
    return issues

def check_directory(directory: Path) -> list:
    """Check all files in directory recursively"""
    all_issues = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            issues = check_file(file_path)
            all_issues.extend(issues)
    
    return all_issues

def main():
    """Main security check function"""
    print("Security Check - Scanning for sensitive information...")
    print("=" * 60)
    
    # Get repository root
    repo_root = Path(__file__).parent.parent
    
    # Check all files
    issues = check_directory(repo_root)
    
    if not issues:
        print("PASS: No sensitive information found!")
        print("Safe to commit.")
        return 0
    
    print(f"FAIL: Found {len(issues)} potential security issues:")
    print()
    
    current_file = None
    for issue in issues:
        if issue['file'] != current_file:
            current_file = issue['file']
            print(f"File: {issue['file']}")
        
        print(f"   Line {issue['line']:3d}: {issue['pattern']}")
        print(f"            {issue['content']}")
        print()
    
    print("=" * 60)
    print("COMMIT BLOCKED - Please remove sensitive information before committing")
    print()
    print("Common fixes:")
    print("- Remove hardcoded file paths")
    print("- Remove case-specific information")
    print("- Use placeholder data in examples")
    print("- Move sensitive data to .env file")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())