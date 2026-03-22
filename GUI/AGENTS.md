# AGENTS.md - Coding Guidelines for YOLO-GUI

## Project Overview

Gradio-based YOLO object detection GUI. Modular architecture with 5 core Python modules using Gradio Blocks for web UI.

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
# or
python gradio_app.py

# Run with hot reload (if available)
gradio gradio_app.py

# Run all tests
pytest tests/ -v

# Run single test
pytest tests/test_module.py::test_function -v

# Run with coverage
pytest --cov=. --cov-report=html

# Type checking
pyright .

# Linting (if ruff configured)
ruff check .
ruff format .

# Start via batch script (Windows)
start.bat
```

## Code Style Guidelines

### File Structure

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Name - Brief description (English)
详细描述 (中文详细说明)
"""

# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# 2. Third-party imports
import numpy as np
import cv2
import gradio as gr
from ultralytics import YOLO

# 3. Local imports
from model_manager import ModelManager
from utils import helper_function

# Module-level constants
DEFAULT_THRESHOLD = 0.25
MODEL_PATHS = [Path("models"), Path("pt_models")]
```

### Naming Conventions

- **Modules**: `snake_case.py` (e.g., `detection_engine.py`)
- **Classes**: `PascalCase` (e.g., `DetectionEngine`, `ModelManager`)
- **Functions**: `snake_case()` with action verbs (e.g., `detect_image()`, `load_model()`)
- **Variables**: `snake_case` (e.g., `conf_threshold`, `model_path`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`, `MODEL_PATHS`)
- **Global variables**: Prefixed with `g_` or clear module-level naming

### Type Hints

Always use type hints for function signatures and important variables:

```python
def detect_image(
    image: np.ndarray, 
    conf_threshold: float = 0.25
) -> DetectionResult:
    """Detect objects in a single image."""
    pass

# For optional types
model: Optional[YOLO] = None
results: List[Dict[str, any]] = []
```

### Error Handling

Use try-except with specific exceptions and meaningful error messages:

```python
try:
    model = YOLO(model_path)
except FileNotFoundError as e:
    raise ValueError(f"Model file not found: {model_path}") from e
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Model loading failed: {str(e)}") from e
```

### Gradio UI Patterns

- Use `gr.Blocks()` for complex layouts
- Organize UI into logical sections with comments
- Name components clearly (e.g., `input_image`, `conf_slider`)
- Use `variant="primary"` for main action buttons
- Add emoji icons to labels for better UX

```python
with gr.Tab("📷 Image Detection"):
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="numpy")
            detect_btn = gr.Button("🔍 Detect", variant="primary")
```

### Documentation

- Bilingual docstrings: English brief + Chinese detailed
- Use Google-style docstrings
- Document parameters, return values, and exceptions

```python
def process_frame(frame: np.ndarray, threshold: float) -> np.ndarray:
    """
    Process a video frame for object detection.
    
    处理视频帧进行目标检测，返回带标注的结果帧。
    
    Args:
        frame: Input video frame in BGR format
        threshold: Detection confidence threshold (0.0-1.0)
        
    Returns:
        Annotated frame with bounding boxes
        
    Raises:
        ValueError: If frame is None or invalid
    """
```

### Testing

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*()`
- Use fixtures in `conftest.py` for shared resources

```python
# tests/test_detection_engine.py
import pytest
from detection_engine import DetectionEngine

@pytest.fixture
def engine():
    return DetectionEngine(mock_model)

def test_detect_image_valid_input(engine):
    result = engine.detect_image(test_image, 0.25)
    assert result.object_count >= 0
    assert result.inference_time > 0
```

### Git Workflow

- Write atomic commits with descriptive messages
- Use semantic prefixes: `feat:`, `fix:`, `docs:`, `refactor:`
- Example: `feat: add batch detection progress bar`

## Project Structure

```
GUI/
├── gradio_app.py          # Main Gradio application
├── model_manager.py       # Model loading and management
├── detection_engine.py    # Core detection logic
├── utils.py              # Helper functions
├── run.py                # Entry point script
├── requirements.txt      # Dependencies
├── start.bat             # Windows launcher
└── tests/                # Test directory
    ├── __init__.py
    ├── conftest.py
    ├── test_model_manager.py
    └── test_detection_engine.py
```

## Dependencies

Core dependencies (from `requirements.txt`):
- `gradio>=4.0.0` - Web UI framework
- `ultralytics>=8.0.0` - YOLO models
- `opencv-python>=4.8.0` - Image/video processing
- `numpy>=1.24.0` - Numerical operations
- `pillow>=10.0.0` - Image utilities

Optional dev dependencies:
- `pytest>=7.0.0` - Testing framework
- `pytest-cov` - Coverage reporting
- `pyright` - Type checking
- `ruff` - Linting and formatting
