# YOLO Web UI - 目标检测Web界面

[![YOLO Object Detection](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/) [![Gradio](https://img.shields.io/badge/UI-Gradio%206.9.0-FF6B6B.svg)](https://gradio.app/)

[English](#english) | [中文](#%E4%B8%AD%E6%96%87)

---

## 🌟 Project Overview

**YOLO Web UI** is a user-friendly web interface for YOLO object detection based on **Gradio 6.9.0**. This project provides a production-ready web interface that allows users to perform object detection through a simple browser interface.

**Note**: This project is only a Web UI frontend and does not include model training functionality. You need to prepare the YOLO model files (.pt format) yourself.

### Features

- 🖼️ **Image Detection** - Upload images for object detection
- 🎬 **Video Processing** - Process video files and generate detection videos
- 📹 **Webcam Detection** - Real-time detection via webcam using Gradio native streaming
- 📂 **Batch Processing** - Process entire folders of images
- 📊 **Multiple Report Formats** - TXT, CSV, and HTML reports with category filtering

---

## 🏗️ Project Structure

```
YOLO_web_UI/
├── GUI/                         # Gradio Web Application
│   ├── gradio_app.py           # Main web interface (Gradio 6.9.0)
│   ├── model_manager.py        # Model discovery and loading
│   ├── detection_engine.py     # Core detection logic
│   ├── utils.py                # Helper utilities
│   ├── run.py                  # Launch script
│   ├── requirements.txt        # Dependencies (Gradio 6.9.0)
│   ├── README.md               # Documentation (Chinese)
│   └── models/                 # Model storage (user-provided)
│
└── models/                     # Alternative model directory
└── pt_models/                  # Alternative model directory
└── weights/                    # Alternative model directory
└── yolo_models/                # Alternative model directory
```

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to GUI directory
cd GUI

# Install dependencies
pip install -r requirements.txt

# Launch the application
python run.py
# Or directly: python gradio_app.py

# Access the interface at: http://localhost:7860
```

### Model Preparation

This project requires YOLO model files (.pt format). You can:

1. **Download pre-trained models** from [Ultralytics Releases](https://github.com/ultralytics/assets/releases)
2. **Use your own trained models**

Place model files in one of these directories:

- `GUI/models/`
- `GUI/pt_models/`
- `GUI/weights/`
- `~/yolo_models/` (user home directory)

**Supported Models**: YOLOv8, YOLOv11, and custom models in .pt format

---

## 🎯 Features

### 📷 Image Detection

- Upload single images for detection
- Adjust confidence threshold (0.01-1.0)
- Visualize bounding boxes with labels
- Download annotated results

### 🎬 Video Processing

- Process video files (.mp4, .avi, .mov)
- Generate detection videos with annotations
- Frame-by-frame analysis
- Progress tracking

### 📹 Real-time Webcam Detection

- Live camera feed processing
- Real-time object detection
- Optimized for performance (configurable FPS)
- Works with built-in or external cameras
- **Powered by**: Gradio 6.x native streaming (no external WebRTC library needed)

### 📂 Batch Processing

- Upload entire folders of images
- Generate comprehensive reports:
  - **TXT Report**: Text summary
  - **CSV Report**: Spreadsheet format for data analysis
  - **HTML Report**: Interactive web report with:
    - Category filtering
    - Hyperlinks to result images
    - Statistical summaries
- Download all results as ZIP

### 🎚️ Advanced Controls

- Model selection from multiple search paths
- Confidence threshold adjustment
- Device selection (CPU/GPU auto-detection)

---

## 🛠️ Technical Specifications

- **Framework**: [Gradio](https://gradio.app/) **6.9.0**
- **Backend**: Python 3.8+
- **UI**: Web-based, responsive design
- **Model Support**: YOLO .pt format (v8, v11, custom)
- **Media Support**: JPG, JPEG, PNG, BMP, WEBP, MP4, AVI, MOV

### Hardware Requirements

- **Minimum**: 4GB RAM, Python 3.8
- **Recommended**: 8GB RAM, CUDA-capable GPU, Python 3.10
- **Storage**: 2GB for installation + model files

---

## 📦 Dependencies

```
gradio==6.9.0        # Web interface framework (Gradio 6.9.0)
ultralytics>=8.0.0   # YOLO model engine
opencv-python>=4.8.0 # Image/video processing
numpy>=1.24.0        # Numerical operations
pillow>=10.0.0       # Image utilities
```

**Note**: No additional WebRTC library (fastrtc/gradio-webrtc) required! Gradio 6.9.0 has built-in streaming support.

---

## 🔧 Usage

### Web Interface

1. **Launch**: `python run.py`
2. **Open browser**: Navigate to `http://localhost:7860`
3. **Load model**: Select a model from the dropdown and click "Load Model"
4. **Select mode**: Image, Video, Webcam, or Batch
5. **Upload media** or enable webcam
6. **Adjust settings**: Confidence threshold
7. **Start detection** and view results
8. **Download** results

### Batch Processing Workflow

1. Go to "📂 Batch Detection" tab
2. Click "Select Folder" and upload a folder containing images
3. Click "Start Batch Detection"
4. Wait for processing to complete
5. Download results:
   - **ZIP**: Complete package with all reports and images
   - **CSV**: Spreadsheet for data analysis
   - **HTML**: Interactive report with category filtering

---

## 🐛 Troubleshooting

### "Model file not found"

**Solution**: Place `.pt` model files in `GUI/models/` or other search paths.

### "CUDA out of memory"

**Solution**: Use smaller model, reduce batch size, or the app will automatically fall back to CPU.

### "Webcam not working"

**Solution**:
- Ensure browser has camera permissions
- Check camera is not in use by other applications
- Try refreshing the page
- The app uses Gradio native streaming, no additional WebRTC library needed

### "ImportError: No module named 'ultralytics'"

**Solution**: Install dependencies:

```bash
cd GUI
pip install -r requirements.txt
```

### Gradio Version Issues

**Solution**: This project requires Gradio 6.9.0. If you have a different version:

```bash
pip install gradio==6.9.0
```

---

## 📄 License

This project follows the license of the original PySide6 project.

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.1.0 | 2026-03-24 | Replaced fastrtc with Gradio 6.x native streaming |
| v2.0.0 | 2026-03-24 | Upgraded to Gradio 6.9.0, optimized for latest API |
| v1.1.0 | 2026-03-24 | Added batch processing with CSV/HTML reports and category filtering |
| v1.0.0 | 2026-03-24 | Initial release with Gradio interface |

---

## 📝 Gradio 6.9.0 Migration Notes

This version has been upgraded to **Gradio 6.9.0** with the following key changes:

1. **CSS/Theme**: Moved from `gr.Blocks(css=...)` to `demo.launch(css=...)`
2. **Event System**: Unified event-driven architecture
3. **Component API**: Standardized component interfaces
4. **Streaming**: Using Gradio 6.x native `gr.Image(streaming=True)` instead of fastrtc WebRTC

### Why Native Streaming?

- **No dependency conflicts**: fastrtc and gradio-webrtc require Gradio <6.0
- **Better compatibility**: Native streaming works seamlessly with Gradio 6.9.0
- **Simpler setup**: No additional WebRTC libraries needed
- **Same functionality**: Real-time webcam detection with comparable performance

---

Made with ❤️ by the open source community

---

---

# YOLO Web UI - 目标检测Web界面

[![YOLO目标检测](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/) [![Gradio](https://img.shields.io/badge/UI-Gradio%206.9.0-FF6B6B.svg)](https://gradio.app/)

[English](#english) | [中文](#%E4%B8%AD%E6%96%87)

---

## 🌟 项目概述

**YOLO Web UI** 是一个基于 **Gradio 6.9.0** 的YOLO目标检测Web界面。本项目提供了一个生产就绪的网页界面，让用户可以通过简单的浏览器界面进行目标检测。

**注意**：本项目仅提供Web UI前端界面，不包含模型训练功能。您需要自行准备YOLO模型文件(.pt格式)。

### 功能特性

- 🖼️ **图片检测** - 上传图片进行目标检测
- 🎬 **视频处理** - 处理视频文件并生成检测视频
- 📹 **摄像头检测** - 通过摄像头进行实时检测（使用Gradio原生streaming）
- 📂 **批量处理** - 处理整个文件夹的图片
- 📊 **多格式报告** - TXT、CSV、HTML报告，支持类别筛选

---

## 🏗️ 项目结构

```
YOLO_web_UI/
├── GUI/                         # Gradio网页应用程序
│   ├── gradio_app.py           # 主网页界面（Gradio 6.9.0）
│   ├── model_manager.py        # 模型发现和加载
│   ├── detection_engine.py     # 核心检测逻辑
│   ├── utils.py                # 辅助工具函数
│   ├── run.py                  # 启动脚本
│   ├── requirements.txt        # 依赖包（Gradio 6.9.0）
│   ├── README.md               # 说明文档（中文）
│   └── models/                 # 模型存储（用户自行提供）
│
└── models/                     # 备选模型目录
└── pt_models/                  # 备选模型目录
└── weights/                    # 备选模型目录
└── yolo_models/                # 备选模型目录
```

---

## 🚀 快速开始

### 安装

```bash
# 进入GUI目录
cd GUI

# 安装依赖包
pip install -r requirements.txt

# 启动应用程序
python run.py
# 或直接运行：python gradio_app.py

# 在浏览器中访问：http://localhost:7860
```

### 模型准备

本项目需要YOLO模型文件(.pt格式)。您可以：

1. **下载预训练模型** 从 [Ultralytics Releases](https://github.com/ultralytics/assets/releases)
2. **使用自己训练的模型**

将模型文件放在以下任一目录：

- `GUI/models/`
- `GUI/pt_models/`
- `GUI/weights/`
- `~/yolo_models/`（用户主目录）

**支持的模型**：YOLOv8、YOLOv11 以及自定义的.pt格式模型

---

## 🎯 功能特性

### 📷 图片检测

- 上传单张图片进行检测
- 调整置信度阈值（0.01-1.0）
- 可视化带标签的边界框
- 下载标注结果

### 🎬 视频处理

- 处理视频文件（.mp4, .avi, .mov）
- 生成带标注的检测视频
- 逐帧分析
- 进度跟踪

### 📹 实时摄像头检测

- 实时摄像头流处理
- 实时目标检测
- 性能优化（可配置FPS）
- 支持内置或外接摄像头
- **技术方案**：使用 Gradio 6.x 原生 streaming（无需外部WebRTC库）

### 📂 批量处理

- 上传整个文件夹的图片
- 生成详细报告：
  - **TXT报告**：文本摘要
  - **CSV报告**：表格格式，便于数据分析
  - **HTML报告**：交互式网页报告，包含：
    - 类别筛选功能
    - 结果图片超链接
    - 统计摘要
- 下载所有结果为ZIP

### 🎚️ 高级控制

- 从多个搜索路径选择模型
- 调整置信度阈值
- 设备选择（CPU/GPU自动检测）

---

## 🛠️ 技术规格

- **框架**: [Gradio](https://gradio.app/) **6.9.0**
- **后端**: Python 3.8+
- **界面**: 基于网页，响应式设计
- **模型支持**: YOLO .pt格式（v8, v11, 自定义）
- **媒体支持**: JPG, JPEG, PNG, BMP, WEBP, MP4, AVI, MOV

### 硬件要求

- **最低**: 4GB内存，Python 3.8
- **推荐**: 8GB内存，支持CUDA的GPU，Python 3.10
- **存储**: 安装需要2GB + 模型文件空间

---

## 📦 依赖包

```
gradio==6.9.0        # 网页界面框架（Gradio 6.9.0）
ultralytics>=8.0.0   # YOLO模型引擎
opencv-python>=4.8.0 # 图片/视频处理
numpy>=1.24.0        # 数值运算
pillow>=10.0.0       # 图片工具
```

**注意**：不需要额外的 WebRTC 库（fastrtc/gradio-webrtc）！Gradio 6.9.0 内置了 streaming 支持。

---

## 🔧 使用方法

### Web界面

1. **启动**: `python run.py`
2. **打开浏览器**: 访问 `http://localhost:7860`
3. **加载模型**: 从下拉菜单选择模型并点击"加载模型"
4. **选择模式**: 图片、视频、摄像头或批量
5. **上传媒体**或启用摄像头
6. **调整设置**: 置信度阈值
7. **开始检测**并查看结果
8. **下载**结果

### 批量处理流程

1. 进入"📂 批量检测"标签页
2. 点击"选择文件夹"并上传包含图片的文件夹
3. 点击"开始批量检测"
4. 等待处理完成
5. 下载结果：
   - **ZIP**: 包含所有报告和图片的完整包
   - **CSV**: 用于数据分析的表格
   - **HTML**: 带类别筛选的交互式报告

---

## 🐛 故障排除

### "找不到模型文件"

**解决方案**: 将`.pt`模型文件放在`GUI/models/`或其他搜索路径中。

### "CUDA内存不足"

**解决方案**: 使用更小的模型、减少批次大小，或应用程序会自动回退到CPU。

### "摄像头无法工作"

**解决方案**:
- 确保浏览器有摄像头权限
- 检查摄像头是否被其他应用程序占用
- 尝试刷新页面
- 本应用使用 Gradio 原生 streaming，无需额外安装 WebRTC 库

### "ImportError: 没有名为'ultralytics'的模块"

**解决方案**: 安装依赖包：

```bash
cd GUI
pip install -r requirements.txt
```

### Gradio版本问题

**解决方案**: 本项目需要 Gradio 6.9.0。如果您使用的是其他版本：

```bash
pip install gradio==6.9.0
```

---

## 📄 许可证

本项目遵循原始PySide6项目的许可证。

---

## 🔄 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.1.0 | 2026-03-24 | 使用 Gradio 6.x 原生 streaming 替代 fastrtc |
| v2.0.0 | 2026-03-24 | 升级到 Gradio 6.9.0，优化适配最新API |
| v1.1.0 | 2026-03-24 | 添加批量处理、CSV/HTML报告、类别筛选 |
| v1.0.0 | 2026-03-24 | 初始版本，Gradio界面 |

---

## 📝 Gradio 6.9.0 迁移说明

本版本已升级到 **Gradio 6.9.0**，主要变更包括：

1. **CSS/主题**: 从 `gr.Blocks(css=...)` 移至 `demo.launch(css=...)`
2. **事件系统**: 统一的事件驱动架构
3. **组件API**: 标准化的组件接口
4. **实时检测**: 使用 Gradio 6.x 原生 `gr.Image(streaming=True)` 替代 fastrtc WebRTC

### 为什么选择原生 Streaming？

- **无依赖冲突**: fastrtc 和 gradio-webrtc 要求 Gradio <6.0
- **更好的兼容性**: 原生 streaming 与 Gradio 6.9.0 无缝集成
- **更简单的安装**: 无需额外的 WebRTC 库
- **相同的功能**: 实时摄像头检测，性能相当

---

由开源社区 ❤️ 制作
