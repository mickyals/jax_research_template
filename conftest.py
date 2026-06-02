import sys
from pathlib import Path

# Make src/ importable for all tests without requiring PYTHONPATH=src
sys.path.insert(0, str(Path(__file__).parent / "src"))
