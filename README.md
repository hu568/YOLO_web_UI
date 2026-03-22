# YOLO-GUI: Complete Object Detection Solution

<div align="center">
  <p>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO Object Detection" width="90%">
  </p>

[English](#english) | [中文](#中文) | [日本語](https://docs.ultralytics.com/ja/) | [한국어](https://docs.ultralytics.com/ko/)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF6B6B.svg)](https://gradio.app/)

</div>

---

<a name="english"></a>
## 🌟 Project Overview

**YOLO-GUI** is a comprehensive object detection platform that combines:
- **GUI Application**: A user-friendly Gradio-based web interface for real-time object detection
- **YOLO Library**: The complete Ultralytics YOLO v8.4.0 engine for training and inference

This project provides both a production-ready web interface and the underlying YOLO library, enabling users to:
- 🖼️ **Detect objects** in images with a simple web interface
- 🎬 **Process videos** with real-time object tracking  
- 📹 **Use webcam** for live detection streams
- 📂 **Batch process** entire directories of images
- 🏋️ **Train custom models** using the included YOLO library
- 🔧 **Extend functionality** with the full Ultralytics API

---

## 🏗️ Project Structure

```
YOLO-GUI/
├── GUI/                         # Gradio Web Application (User Interface)
│   ├── gradio_app.py           # Main web interface (448 lines)
│   ├── model_manager.py        # Model discovery and loading
│   ├── detection_engine.py     # Core detection logic
│   ├── utils.py                # Helper utilities
│   ├── run.py                  # Launch script with dependency checks
│   ├── start.bat               # Windows launcher
│   ├── requirements.txt        # GUI dependencies
│   ├── README.md               # GUI documentation (Chinese)
│   ├── AGENTS.md               # GUI development guidelines
│   └── models/                 # Pre-trained model storage
│
└── YOLO/                       # Ultralytics YOLO Library v8.4.0
    ├── ultralytics/            # Main Python package
    ├── tests/                  # Comprehensive test suite
    ├── examples/               # 20+ integration examples
    ├── docs/                   # Full documentation
    ├── pyproject.toml          # Package configuration
    ├── README.md               # YOLO library documentation
    ├── LICENSE                 # AGPL-3.0 license
    └── CONTRIBUTING.md         # Contribution guidelines
```

---

## 🚀 Quick Start

### Option 1: Use the Web Interface (Recommended for Users)

```bash
# Navigate to GUI directory
cd GUI

# Install dependencies
pip install -r requirements.txt

# Launch the application
python run.py
# Or directly: python gradio_app.py
# Or use Windows launcher: start.bat

# Access the interface at: http://localhost:7860
```

### Option 2: Use the YOLO Library (Recommended for Developers)

```bash
# Navigate to YOLO directory
cd YOLO

# Install the YOLO library
pip install -e .
# Or install from PyPI: pip install ultralytics

# Run object detection on an image
yolo predict model=yolo26n.pt source='path/to/image.jpg'

# Train a custom model
yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
```

---

## 🎯 GUI Features

### 📷 **Image Detection**
- Upload single or multiple images
- Adjust confidence threshold (0.01-1.0)
- Visualize bounding boxes with labels
- Download annotated results

### 🎬 **Video Processing**
- Process video files (.mp4, .avi, .mov)
- Generate detection videos
- Frame-by-frame analysis
- Export results with timestamps

### 📹 **Real-time Webcam Detection**
- Live camera feed processing
- Real-time object tracking
- Low-latency inference
- Works with built-in or external cameras

### 📂 **Batch Processing**
- Process entire directories of images
- Generate comprehensive reports
- Batch statistics and summaries
- Organized output structure

### 🎚️ **Advanced Controls**
- Model selection from multiple search paths
- Confidence threshold adjustment
- NMS (Non-Maximum Suppression) settings
- Device selection (CPU/GPU)

---

## 🛠️ Technical Specifications

### GUI Application
- **Framework**: [Gradio](https://gradio.app/) v4.0+
- **Backend**: Python 3.8+
- **UI**: Web-based with responsive design
- **Model Support**: YOLO .pt format (v8, v11, custom)
- **Media Support**: JPG, PNG, MP4, AVI, Webcam

### YOLO Library
- **Version**: Ultralytics YOLO v8.4.0
- **Tasks**: Detection, Segmentation, Classification, Pose Estimation, OBB
- **Models**: YOLO26n, YOLO26s, YOLO26m, YOLO26l, YOLO26x
- **Export Formats**: ONNX, TensorRT, OpenVINO, CoreML, TensorFlow
- **Training**: Distributed, multi-GPU, hyperparameter evolution

### Hardware Requirements
- **Minimum**: 4GB RAM, 2GB VRAM (for GPU), Python 3.8
- **Recommended**: 8GB RAM, 6GB VRAM, CUDA 11.8, Python 3.10
- **Storage**: 2GB for installation, + space for models

---

## 📦 Installation Details

### Full Installation (GUI + YOLO)

```bash
# Clone the repository
git clone <repository-url>
cd YOLO-GUI

# Install GUI dependencies
cd GUI
pip install -r requirements.txt

# Install YOLO library (editable mode)
cd ../YOLO
pip install -e .

# Return to GUI and launch
cd ../GUI
python run.py
```

### Dependencies

**GUI Application (`GUI/requirements.txt`):**
```txt
gradio>=4.0.0        # Web interface framework
ultralytics>=8.0.0   # YOLO model engine
opencv-python>=4.8.0 # Image/video processing
numpy>=1.24.0        # Numerical operations
pillow>=10.0.0       # Image utilities
```

**YOLO Library (`YOLO/pyproject.toml`):**
- Core: PyTorch, torchvision, opencv-python, numpy
- Optional: ONNX, TensorRT, OpenVINO, TensorFlow exports
- Development: pytest, ruff, mypy, mkdocs

---

## 🔧 Usage Examples

### Using the GUI Interface

1. **Launch the application**: `python run.py`
2. **Open browser**: Navigate to `http://localhost:7860`
3. **Select detection mode**: Image, Video, Webcam, or Batch
4. **Upload media** or enable webcam
5. **Adjust settings**: Model, confidence threshold
6. **Start detection** and view results
7. **Download** or save annotated outputs

### Using the YOLO CLI

```bash
# Detect objects in an image
yolo predict model=yolo26n.pt source='bus.jpg'

# Detect in a video
yolo predict model=yolo26n.pt source='video.mp4'

# Train on custom data
yolo detect train data=custom.yaml model=yolo26n.pt epochs=50

# Export to ONNX format
yolo export model=yolo26n.pt format=onnx

# Validate model performance
yolo val model=yolo26n.pt data=coco.yaml
```

### Using the YOLO Python API

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo26n.pt")

# Predict on an image
results = model("image.jpg")

# Train on custom dataset
model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Export model
model.export(format="onnx")
```

---

## 📁 Model Management

### Default Model Search Paths
The GUI automatically searches for `.pt` models in:
- `GUI/models/`
- `GUI/pt_models/`
- `GUI/weights/`
- `GUI/yolo_models/`
- `~/yolo_models/` (user home directory)
- `C:/yolo_models/` (Windows)
- `/usr/local/share/yolo_models/` (Linux)

### Pre-trained Models
- **YOLO26n**: 2.4M params, 40.9 mAP, fastest inference
- **YOLO26s**: 9.5M params, 48.6 mAP, balanced performance
- **YOLO26m**: 20.4M params, 53.1 mAP, high accuracy
- **YOLO26l**: 24.8M params, 55.0 mAP, professional grade
- **YOLO26x**: 55.7M params, 57.5 mAP, state-of-the-art

Download models from [Ultralytics Releases](https://github.com/ultralytics/assets/releases) and place in any search path.

---

## 🧩 Integration Examples

The YOLO directory includes 20+ integration examples:

| Integration | Directory | Description |
|------------|-----------|-------------|
| **ONNXRuntime** | `examples/YOLOv8-ONNXRuntime/` | Cross-platform inference |
| **TensorRT** | `examples/YOLO11-Triton-CPP/` | GPU-accelerated inference |
| **OpenVINO** | `examples/YOLOv8-OpenVINO-CPP-Inference/` | Intel hardware acceleration |
| **TensorFlow** | `examples/YOLOv8-TFLite-Python/` | Mobile deployment |
| **CoreML** | `examples/` | Apple ecosystem deployment |
| **OpenCV** | `examples/YOLOv8-OpenCV-ONNX-Python/` | Traditional CV pipeline |

---

## 🔍 Development Guidelines

### Code Style
- **Modules**: `snake_case.py` (e.g., `detection_engine.py`)
- **Classes**: `PascalCase` (e.g., `DetectionEngine`)
- **Functions**: `snake_case()` (e.g., `detect_image()`)
- **Variables**: `snake_case` (e.g., `conf_threshold`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`)

### Type Hints
```python
def detect_image(image: np.ndarray, conf_threshold: float = 0.25) -> DetectionResult:
    """Detect objects in a single image."""
    pass
```

### Error Handling
```python
try:
    model = YOLO(model_path)
except FileNotFoundError as e:
    raise ValueError(f"Model file not found: {model_path}") from e
```

### Documentation Style
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

---

## 🧪 Testing

### GUI Application Testing
```bash
cd GUI
pytest tests/ -v
pytest --cov=. --cov-report=html
pyright .  # Type checking
```

### YOLO Library Testing
```bash
cd YOLO
pytest
pytest --slow  # Include slow tests
pytest --cov=ultralytics
pytest --durations=30 --color=yes
```

### Linting and Formatting
```bash
# YOLO library
ruff check .
ruff format .
yapf -i -r .
isort .
codespell

# GUI application (when configured)
pyright .
```

---

## 📄 License

### YOLO Library
- **Primary License**: [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0)
- **Commercial Use**: Requires [Ultralytics Enterprise License](https://www.ultralytics.com/license)
- **Open Source**: Free for research, education, personal use

### GUI Application
- Follows licensing of original PySide6 project
- Compatible with YOLO's AGPL-3.0 license
- Contact project maintainers for commercial licensing

---

## 🤝 Contributing

We welcome contributions! Please see:

1. **[YOLO/CONTRIBUTING.md](YOLO/CONTRIBUTING.md)** - Contribution guidelines for YOLO library
2. **[AGENTS.md](AGENTS.md)** - Development guidelines for both projects
3. **[GUI/AGENTS.md](GUI/AGENTS.md)** - GUI-specific coding standards

### Commit Guidelines
- Use semantic prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `style:`
- Write atomic commits with descriptive messages
- Example: `feat: add batch detection progress bar to GUI`

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request with detailed description

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Model file not found"
**Solution**: Place `.pt` model files in one of the search paths listed above, or specify full path.

#### 2. "CUDA out of memory"
**Solution**: Reduce batch size, use smaller model (YOLO26n), or disable GPU:
```bash
yolo predict model=yolo26n.pt source='image.jpg' device=cpu
```

#### 3. "Webcam not working"
**Solution**: 
- Ensure browser has camera permissions
- Check camera is not in use by other applications
- Try different camera index (0, 1, 2...)

#### 4. "ImportError: No module named 'ultralytics'"
**Solution**: Install the YOLO library:
```bash
cd YOLO
pip install -e .
# Or: pip install ultralytics
```

#### 5. "Video processing too slow"
**Solution**:
- Reduce video resolution
- Skip frames with `skip_frames` parameter
- Use GPU acceleration if available
- Choose lighter model (YOLO26n instead of YOLO26x)

---

## 📚 Documentation Links

### YOLO Library Documentation
- **[Official Docs](https://docs.ultralytics.com/)** - Complete YOLO documentation
- **[Python API](https://docs.ultralytics.com/usage/python/)** - Python usage examples
- **[CLI Reference](https://docs.ultralytics.com/usage/cli/)** - Command-line interface
- **[Models](https://docs.ultralytics.com/models/)** - Model architectures and benchmarks
- **[Tasks](https://docs.ultralytics.com/tasks/)** - Detection, segmentation, classification

### GUI Documentation
- **[GUI/README.md](GUI/README.md)** - Detailed GUI documentation (Chinese)
- **[GUI/AGENTS.md](GUI/AGENTS.md)** - GUI development guidelines

### Community Resources
- **[GitHub Issues](https://github.com/ultralytics/ultralytics/issues)** - Bug reports and feature requests
- **[Discord](https://discord.com/invite/ultralytics)** - Community discussions
- **[Ultralytics Forums](https://community.ultralytics.com/)** - Technical support

---

## 📧 Contact & Support

### Project Maintainers
- **GUI Application**: Original PySide6 project maintainers
- **YOLO Library**: [Ultralytics Team](https://www.ultralytics.com/)

### Getting Help
1. **GitHub Issues**: For bugs and feature requests
2. **Documentation**: Check [docs.ultralytics.com](https://docs.ultralytics.com/)
3. **Community**: Join [Discord](https://discord.com/invite/ultralytics)
4. **Enterprise**: Contact [Ultralytics Licensing](https://www.ultralytics.com/license)

### Citation
If you use YOLO in your research, please cite:
```bibtex
@software{yolov8_ultralytics,
  title = {Ultralytics YOLOv8},
  author = {Jocher, Glenn and Qiu, Jing and others},
  year = {2023},
  version = {8.0.0},
  license = {AGPL-3.0},
  url = {https://github.com/ultralytics/ultralytics}
}
```

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0.0 | 2026-03-20 | Initial release with GUI + YOLO v8.4.0 |
| GUI v1.0 | 2026-03-20 | Gradio interface with image/video/webcam/batch detection |
| YOLO v8.4.0 | 2026-03-20 | Ultralytics YOLO library with YOLO26 models |

---

<div align="center">
  <p>Made with ❤️ by the open source community</p>
  
  <a href="https://github.com/ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="GitHub">
  </a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://discord.com/invite/ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Discord">
  </a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.youtube.com/@ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="YouTube">
  </a>
</div>

---

<a name="中文"></a>
# YOLO-GUI: 完整的目标检测解决方案

<div align="center">
  <p>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO目标检测" width="90%">
  </p>

[English](#english) | [中文](#中文) | [日本語](https://docs.ultralytics.com/ja/) | [한국어](https://docs.ultralytics.com/ko/)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF6B6B.svg)](https://gradio.app/)

</div>

---

## 🌟 项目概述

**YOLO-GUI** 是一个完整的目标检测平台，包含：
- **GUI应用程序**: 基于Gradio的用户友好网页界面，用于实时目标检测
- **YOLO库**: 完整的Ultralytics YOLO v8.4.0引擎，用于训练和推理

本项目提供了一个生产就绪的网页界面和底层的YOLO库，用户可以：
- 🖼️ **检测图片**中的物体，通过简单的网页界面
- 🎬 **处理视频**，带实时物体追踪
- 📹 **使用摄像头**进行实时检测流
- 📂 **批量处理**整个图片目录
- 🏋️ **训练自定义模型**，使用内置的YOLO库
- 🔧 **扩展功能**，使用完整的Ultralytics API

---

## 🏗️ 项目结构

```
YOLO-GUI/
├── GUI/                         # Gradio网页应用程序（用户界面）
│   ├── gradio_app.py           # 主网页界面（448行）
│   ├── model_manager.py        # 模型发现和加载
│   ├── detection_engine.py     # 核心检测逻辑
│   ├── utils.py                # 辅助工具函数
│   ├── run.py                  # 启动脚本（含依赖检查）
│   ├── start.bat               # Windows启动器
│   ├── requirements.txt        # GUI依赖包
│   ├── README.md               # GUI文档（中文）
│   ├── AGENTS.md               # GUI开发指南
│   └── models/                 # 预训练模型存储
│
└── YOLO/                       # Ultralytics YOLO库 v8.4.0
    ├── ultralytics/            # 主Python包
    ├── tests/                  # 完整的测试套件
    ├── examples/               # 20+集成示例
    ├── docs/                   # 完整文档
    ├── pyproject.toml          # 包配置
    ├── README.md               # YOLO库文档
    ├── LICENSE                 # AGPL-3.0许可证
    └── CONTRIBUTING.md         # 贡献指南
```

---

## 🚀 快速开始

### 选项1：使用网页界面（推荐给用户）

```bash
# 进入GUI目录
cd GUI

# 安装依赖包
pip install -r requirements.txt

# 启动应用程序
python run.py
# 或直接运行：python gradio_app.py
# 或使用Windows启动器：start.bat

# 在浏览器中访问：http://localhost:7860
```

### 选项2：使用YOLO库（推荐给开发者）

```bash
# 进入YOLO目录
cd YOLO

# 安装YOLO库
pip install -e .
# 或从PyPI安装：pip install ultralytics

# 在图片上运行目标检测
yolo predict model=yolo26n.pt source='path/to/image.jpg'

# 训练自定义模型
yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
```

---

## 🎯 GUI功能

### 📷 **图片检测**
- 上传单张或多张图片
- 调整置信度阈值（0.01-1.0）
- 可视化带标签的边界框
- 下载标注结果

### 🎬 **视频处理**
- 处理视频文件（.mp4, .avi, .mov）
- 生成检测视频
- 逐帧分析
- 导出带时间戳的结果

### 📹 **实时摄像头检测**
- 实时摄像头流处理
- 实时物体追踪
- 低延迟推理
- 支持内置或外接摄像头

### 📂 **批量处理**
- 处理整个目录的图片
- 生成详细报告
- 批量统计和摘要
- 组织好的输出结构

### 🎚️ **高级控制**
- 从多个搜索路径选择模型
- 调整置信度阈值
- NMS（非极大值抑制）设置
- 设备选择（CPU/GPU）

---

## 🛠️ 技术规格

### GUI应用程序
- **框架**: [Gradio](https://gradio.app/) v4.0+
- **后端**: Python 3.8+
- **界面**: 基于网页，响应式设计
- **模型支持**: YOLO .pt格式（v8, v11, 自定义）
- **媒体支持**: JPG, PNG, MP4, AVI, 摄像头

### YOLO库
- **版本**: Ultralytics YOLO v8.4.0
- **任务**: 检测、分割、分类、姿态估计、OBB
- **模型**: YOLO26n, YOLO26s, YOLO26m, YOLO26l, YOLO26x
- **导出格式**: ONNX, TensorRT, OpenVINO, CoreML, TensorFlow
- **训练**: 分布式、多GPU、超参数进化

### 硬件要求
- **最低**: 4GB内存，2GB显存（GPU），Python 3.8
- **推荐**: 8GB内存，6GB显存，CUDA 11.8，Python 3.10
- **存储**: 安装需要2GB，+ 模型存储空间

---

## 📦 安装详情

### 完整安装（GUI + YOLO）

```bash
# 克隆仓库
git clone <仓库地址>
cd YOLO-GUI

# 安装GUI依赖
cd GUI
pip install -r requirements.txt

# 安装YOLO库（可编辑模式）
cd ../YOLO
pip install -e .

# 返回GUI并启动
cd ../GUI
python run.py
```

### 依赖包

**GUI应用程序 (`GUI/requirements.txt`):**
```txt
gradio>=4.0.0        # 网页界面框架
ultralytics>=8.0.0   # YOLO模型引擎
opencv-python>=4.8.0 # 图片/视频处理
numpy>=1.24.0        # 数值运算
pillow>=10.0.0       # 图片工具
```

**YOLO库 (`YOLO/pyproject.toml`):**
- 核心: PyTorch, torchvision, opencv-python, numpy
- 可选: ONNX, TensorRT, OpenVINO, TensorFlow导出
- 开发: pytest, ruff, mypy, mkdocs

---

## 🔧 使用示例

### 使用GUI界面

1. **启动应用程序**: `python run.py`
2. **打开浏览器**: 访问 `http://localhost:7860`
3. **选择检测模式**: 图片、视频、摄像头或批量
4. **上传媒体**或启用摄像头
5. **调整设置**: 模型、置信度阈值
6. **开始检测**并查看结果
7. **下载**或保存标注输出

### 使用YOLO命令行

```bash
# 在图片中检测物体
yolo predict model=yolo26n.pt source='bus.jpg'

# 在视频中检测
yolo predict model=yolo26n.pt source='video.mp4'

# 在自定义数据上训练
yolo detect train data=custom.yaml model=yolo26n.pt epochs=50

# 导出为ONNX格式
yolo export model=yolo26n.pt format=onnx

# 验证模型性能
yolo val model=yolo26n.pt data=coco.yaml
```

### 使用YOLO Python API

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo26n.pt")

# 在图片上预测
results = model("image.jpg")

# 在自定义数据集上训练
model.train(data="coco8.yaml", epochs=100, imgsz=640)

# 导出模型
model.export(format="onnx")
```

---

## 📁 模型管理

### 默认模型搜索路径
GUI自动在以下路径搜索`.pt`模型：
- `GUI/models/`
- `GUI/pt_models/`
- `GUI/weights/`
- `GUI/yolo_models/`
- `~/yolo_models/`（用户主目录）
- `C:/yolo_models/`（Windows）
- `/usr/local/share/yolo_models/`（Linux）

### 预训练模型
- **YOLO26n**: 2.4M参数，40.9 mAP，最快推理速度
- **YOLO26s**: 9.5M参数，48.6 mAP，平衡性能
- **YOLO26m**: 20.4M参数，53.1 mAP，高精度
- **YOLO26l**: 24.8M参数，55.0 mAP，专业级
- **YOLO26x**: 55.7M参数，57.5 mAP，最先进水平

从[Ultralytics Releases](https://github.com/ultralytics/assets/releases)下载模型，放在任意搜索路径中。

---

## 🧩 集成示例

YOLO目录包含20+集成示例：

| 集成 | 目录 | 描述 |
|------|------|------|
| **ONNXRuntime** | `examples/YOLOv8-ONNXRuntime/` | 跨平台推理 |
| **TensorRT** | `examples/YOLO11-Triton-CPP/` | GPU加速推理 |
| **OpenVINO** | `examples/YOLOv8-OpenVINO-CPP-Inference/` | Intel硬件加速 |
| **TensorFlow** | `examples/YOLOv8-TFLite-Python/` | 移动端部署 |
| **CoreML** | `examples/` | Apple生态系统部署 |
| **OpenCV** | `examples/YOLOv8-OpenCV-ONNX-Python/` | 传统CV管道 |

---

## 🔍 开发指南

### 代码风格
- **模块**: `snake_case.py`（如：`detection_engine.py`）
- **类**: `PascalCase`（如：`DetectionEngine`）
- **函数**: `snake_case()`（如：`detect_image()`）
- **变量**: `snake_case`（如：`conf_threshold`）
- **常量**: `UPPER_SNAKE_CASE`（如：`DEFAULT_THRESHOLD`）

### 类型提示
```python
def detect_image(image: np.ndarray, conf_threshold: float = 0.25) -> DetectionResult:
    """在单张图片中检测物体。"""
    pass
```

### 错误处理
```python
try:
    model = YOLO(model_path)
except FileNotFoundError as e:
    raise ValueError(f"找不到模型文件: {model_path}") from e
```

### 文档风格
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

---

## 🧪 测试

### GUI应用程序测试
```bash
cd GUI
pytest tests/ -v
pytest --cov=. --cov-report=html
pyright .  # 类型检查
```

### YOLO库测试
```bash
cd YOLO
pytest
pytest --slow  # 包含慢速测试
pytest --cov=ultralytics
pytest --durations=30 --color=yes
```

### 代码检查和格式化
```bash
# YOLO库
ruff check .
ruff format .
yapf -i -r .
isort .
codespell

# GUI应用程序（配置后）
pyright .
```

---

## 📄 许可证

### YOLO库
- **主要许可证**: [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0)
- **商业使用**: 需要[Ultralytics企业许可证](https://www.ultralytics.com/license)
- **开源**: 研究、教育、个人使用免费

### GUI应用程序
- 遵循原始PySide6项目的许可证
- 与YOLO的AGPL-3.0许可证兼容
- 商业使用请联系项目维护者

---

## 🤝 贡献

我们欢迎贡献！请查看：

1. **[YOLO/CONTRIBUTING.md](YOLO/CONTRIBUTING.md)** - YOLO库的贡献指南
2. **[AGENTS.md](AGENTS.md)** - 两个项目的开发指南
3. **[GUI/AGENTS.md](GUI/AGENTS.md)** - GUI特定的编码标准

### 提交指南
- 使用语义前缀：`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `style:`
- 编写描述性消息的原子提交
- 示例：`feat: 为GUI添加批量检测进度条`

### Pull Request流程
1. Fork仓库
2. 创建特性分支
3. 为新功能添加测试
4. 确保所有测试通过
5. 提交详细的pull request描述

---

## 🐛 故障排除

### 常见问题

#### 1. "找不到模型文件"
**解决方案**: 将`.pt`模型文件放在上述任一搜索路径中，或指定完整路径。

#### 2. "CUDA内存不足"
**解决方案**: 减小批次大小，使用更小的模型（YOLO26n），或禁用GPU：
```bash
yolo predict model=yolo26n.pt source='image.jpg' device=cpu
```

#### 3. "摄像头无法工作"
**解决方案**: 
- 确保浏览器有摄像头权限
- 检查摄像头是否被其他应用程序占用
- 尝试不同的摄像头索引（0, 1, 2...）

#### 4. "ImportError: 没有名为'ultralytics'的模块"
**解决方案**: 安装YOLO库：
```bash
cd YOLO
pip install -e .
# 或：pip install ultralytics
```

#### 5. "视频处理太慢"
**解决方案**:
- 降低视频分辨率
- 使用`skip_frames`参数跳过帧
- 如果可用，使用GPU加速
- 选择更轻量的模型（用YOLO26n代替YOLO26x）

---

## 📚 文档链接

### YOLO库文档
- **[官方文档](https://docs.ultralytics.com/)** - 完整的YOLO文档
- **[Python API](https://docs.ultralytics.com/usage/python/)** - Python使用示例
- **[CLI参考](https://docs.ultralytics.com/usage/cli/)** - 命令行界面
- **[模型](https://docs.ultralytics.com/models/)** - 模型架构和基准
- **[任务](https://docs.ultralytics.com/tasks/)** - 检测、分割、分类

### GUI文档
- **[GUI/README.md](GUI/README.md)** - 详细的GUI文档（中文）
- **[GUI/AGENTS.md](GUI/AGENTS.md)** - GUI开发指南

### 社区资源
- **[GitHub Issues](https://github.com/ultralytics/ultralytics/issues)** - 错误报告和功能请求
- **[Discord](https://discord.com/invite/ultralytics)** - 社区讨论
- **[Ultralytics论坛](https://community.ultralytics.com/)** - 技术支持

---

## 📧 联系与支持

### 项目维护者
- **GUI应用程序**: 原始PySide6项目维护者
- **YOLO库**: [Ultralytics团队](https://www.ultralytics.com/)

### 获取帮助
1. **GitHub Issues**: 用于错误报告和功能请求
2. **文档**: 查看[docs.ultralytics.com](https://docs.ultralytics.com/)
3. **社区**: 加入[Discord](https://discord.com/invite/ultralytics)
4. **企业版**: 联系[Ultralytics许可](https://www.ultralytics.com/license)

### 引用
如果在研究中使用YOLO，请引用：
```bibtex
@software{yolov8_ultralytics,
  title = {Ultralytics YOLOv8},
  author = {Jocher, Glenn and Qiu, Jing and others},
  year = {2023},
  version = {8.0.0},
  license = {AGPL-3.0},
  url = {https://github.com/ultralytics/ultralytics}
}
```

---

## 🔄 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0.0 | 2026-03-20 | 初始版本，包含GUI + YOLO v8.4.0 |
| GUI v1.0 | 2026-03-20 | Gradio界面，支持图片/视频/摄像头/批量检测 |
| YOLO v8.4.0 | 2026-03-20 | Ultralytics YOLO库，包含YOLO26模型 |

---

<div align="center">
  <p>由开源社区 ❤️ 制作</p>
  
  <a href="https://github.com/ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="GitHub">
  </a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://discord.com/invite/ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Discord">
  </a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.youtube.com/@ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="YouTube">
  </a>
</div>