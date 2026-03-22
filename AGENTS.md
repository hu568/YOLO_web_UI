# AGENTS.md - Development Guidelines for YOLO-GUI

## Project Structure Overview

This repository contains two main projects:

1. **GUI/**: Gradio-based YOLO object detection GUI application
   - Python application with Gradio UI for YOLO detection
   - Core modules: `gradio_app.py`, `model_manager.py`, `detection_engine.py`, `utils.py`
   - Already has comprehensive AGENTS.md at `GUI/AGENTS.md`

2. **YOLO/**: Ultralytics YOLO library (version v8.4.0)
   - Complete YOLO implementation library (AGPL-3.0 licensed)
   - Core ML library with training, inference, export capabilities
   - Standard Ultralytics project structure with tests

## Quick Commands

### GUI Application (Python Gradio App)

```bash
# Install dependencies
cd GUI
pip install -r requirements.txt

# Run application
python run.py                   # Recommended (checks dependencies)
python gradio_app.py           # Direct launch
gradio gradio_app.py           # With hot reload (if available)
./start.bat                    # Windows batch launcher

# Testing (when tests are added)
pytest tests/ -v
pytest tests/test_module.py::test_function -v
pytest --cov=. --cov-report=html

# Type checking
pyright .
```

### YOLO Library (Ultralytics Development)

```bash
# Development installation
cd YOLO
pip install -e .               # Editable install
pip install -e ".[dev]"        # With dev tools

# Testing
pytest                         # Run all tests
pytest --slow                  # Include slow tests  
pytest --cov=ultralytics       # With coverage
pytest --durations=30 --color=yes

# Linting & Formatting
ruff check .                   # Lint with ruff
ruff format .                  # Format with ruff
yapf -i -r .                   # Alternative formatter
isort .                        # Sort imports
codespell                      # Spell checking

# YOLO CLI commands
yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
yolo predict model=yolo26n.pt source='path/to/image.jpg'
```

## Code Style Guidelines

### Import Order (Consistent Across Projects)

```python
# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# 2. Third-party imports  
import numpy as np
import cv2
import gradio as gr
import torch
from ultralytics import YOLO

# 3. Local imports
from model_manager import ModelManager
from utils import helper_function
```

### Naming Conventions

- **Modules**: `snake_case.py` (e.g., `detection_engine.py`, `model_manager.py`)
- **Classes**: `PascalCase` (e.g., `DetectionEngine`, `Model`, `YOLODataset`)
- **Functions/Methods**: `snake_case()` with action verbs (e.g., `detect_image()`, `load_model()`)
- **Variables**: `snake_case` (e.g., `conf_threshold`, `model_path`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`, `ASSETS`)
- **Private methods**: `_leading_underscore` (e.g., `_load_config()`, `_check_is_pytorch_model()`)

### Type Hints (Modern Python)

```python
# GUI style (explicit typing)
from typing import Optional, List, Dict

def detect_image(image: np.ndarray, conf_threshold: float = 0.25) -> DetectionResult:
    """Detect objects in a single image."""
    pass

# YOLO style (PEP 585 with __future__ import)
from __future__ import annotations

def __init__(self, model: str | Path | Model = "yolo26n.pt", task: str | None = None):
    """Initialize model with modern typing syntax."""
    pass
```

### Error Handling Patterns

**GUI Pattern (User-friendly messages):**
```python
try:
    model = YOLO(model_path)
except FileNotFoundError as e:
    raise ValueError(f"Model file not found: {model_path}") from e
except Exception as e:
    print(f"模型加载失败: {e}")
    raise e
```

**YOLO Pattern (Logging with custom exceptions):**
```python
try:
    return tuple(map(int, re.findall(r"\d+", version)[:3]))
except Exception as e:
    LOGGER.warning(f"failure for parse_version({version}), returning (0, 0, 0): {e}")
    return 0, 0, 0
```

### Documentation Style

**Bilingual Docstrings (GUI):**
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

**Google-style Docstrings (YOLO):**
```python
def __init__(self, model: str | Path, task: str | None = None) -> None:
    """Initialize a new instance of the YOLO model class.

    Args:
        model (str | Path | Model): Path or name of the model to load or create.
        task (str, optional): The specific task for the model.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
        ValueError: If the model file or configuration is invalid.
    """
```

## Project-Specific Guidelines

### GUI Application (`GUI/`)
- Use `gr.Blocks()` for complex UI layouts
- Organize UI with comments and logical sections  
- Add emoji icons for better UX (e.g., `"📷 图片检测"`)
- Name components clearly (e.g., `input_image`, `conf_slider`)
- Use `variant="primary"` for main action buttons
- Keep UI responsive with progress bars for long operations

### YOLO Library (`YOLO/`)
- All files start with license header: `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license`
- Line length: 120 characters (not 80)
- Use `pathlib.Path` exclusively (not `os.path`)
- Prefer list/dict comprehensions over `map()`/`filter()`
- Use f-strings for formatting
- Maintain backward compatibility for public APIs

## Missing Configuration Notes

No Cursor IDE rules (`.cursorrules`, `.cursor/rules/`) or GitHub Copilot instructions (`.github/copilot-instructions.md`) were found. Consider adding:
- `.cursorrules` for Cursor AI assistant
- `.github/copilot-instructions.md` for GitHub Copilot

## Important Notes for AI Agents

1. **Project Context**: The GUI application depends on the YOLO library for detection capabilities
2. **License Compliance**: YOLO code is AGPL-3.0 licensed - respect licensing in modifications
3. **Testing**: GUI needs test suite (`tests/`) - currently missing
4. **Linting**: GUI lacks lint config (`ruff.toml`, `pyproject.toml`) - YOLO has comprehensive setup
5. **Documentation**: Follow existing patterns - bilingual for GUI, Google-style for YOLO
6. **Error Messages**: GUI should have user-friendly Chinese/English messages, YOLO uses logging

## Git Workflow

- Write atomic commits with descriptive messages
- Use semantic prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `style:`
- Example: `feat: add batch detection progress bar to GUI`
- Reference existing PR guidelines in `YOLO/CONTRIBUTING.md`

---
*Generated from analysis of E:\YOLO-GUI on 2026-03-20*