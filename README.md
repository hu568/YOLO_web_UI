# YOLO Web UI - Gradio 6.9.0

使用 Gradio 6.9.0 重构的 YOLO 目标检测 Web 界面。

## 🚀 快速开始

```bash
cd GUI
pip install -r requirements.txt
python run.py
```

然后访问 http://localhost:7860

## 📁 项目结构

```
YOLO_web_UI/
├── GUI/                    # 主应用程序
│   ├── gradio_app.py      # Gradio 6.9.0 主界面
│   ├── model_manager.py   # 模型管理
│   ├── detection_engine.py # 检测引擎
│   ├── utils.py           # 工具函数
│   ├── run.py             # 启动脚本
│   ├── requirements.txt   # 依赖（Gradio 6.9.0）
│   └── README.md          # 详细文档
├── models/                # 模型目录
├── pt_models/            # 模型目录
├── weights/              # 模型目录
└── yolo_models/          # 模型目录
```

## ✨ 功能特性

- 📷 **图片检测** - 单张图片目标检测
- 🎬 **视频检测** - 视频文件处理
- 📹 **实时检测** - WebRTC 摄像头实时检测
- 📂 **批量检测** - 文件夹批量处理，生成 TXT/CSV/HTML 报告
- 📊 **类别筛选** - HTML 报告支持交互式类别筛选

## 📝 Gradio 6.9.0 升级说明

本次重构主要适配 Gradio 6.9.0 的 API 变更：

1. **CSS/主题配置**: 从 `gr.Blocks(css=...)` 移至 `demo.launch(css=...)`
2. **事件系统**: 统一的事件驱动架构
3. **组件接口**: 标准化的组件定义方式
4. **依赖更新**: `requirements.txt` 中指定 `gradio==6.9.0`

## 📦 依赖

```
gradio==6.9.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
fastrtc>=0.1.0  # 可选，用于实时检测
```

## 📄 许可证

遵循原始项目许可证。

---

重构完成日期: 2026-03-24
