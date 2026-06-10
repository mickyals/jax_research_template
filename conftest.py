import sys
from pathlib import Path

# Make jrt/ importable for all tests without requiring PYTHONPATH=jrt
sys.path.insert(0, str(Path(__file__).parent / "jrt"))
