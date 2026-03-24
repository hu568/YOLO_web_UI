#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Script - 启动脚本
启动Gradio YOLO检测应用
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """检查必要的依赖"""
    missing = []

    try:
        import gradio

        print(f"✅ Gradio 版本: {gradio.__version__}")
    except ImportError:
        missing.append("gradio")

    try:
        import ultralytics

        print(f"✅ Ultralytics 已安装")
    except ImportError:
        missing.append("ultralytics")

    try:
        import cv2

        print(f"✅ OpenCV 版本: {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy

        print(f"✅ NumPy 版本: {numpy.__version__}")
    except ImportError:
        missing.append("numpy")

    try:
        from PIL import Image

        print(f"✅ Pillow 已安装")
    except ImportError:
        missing.append("pillow")

    if missing:
        print("\n❌ 缺少以下依赖包:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n请运行: pip install -r requirements.txt")
        return False

    return True


def create_directories():
    """创建必要的目录"""
    dirs = ["models", "pt_models", "weights", "yolo_models"]

    for d in dirs:
        Path(d).mkdir(exist_ok=True)

    print("✅ 目录检查完成")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 YOLO目标检测系统 - Gradio版本")
    print("=" * 60)
    print()

    # 检查依赖
    print("🔍 检查依赖...")
    if not check_dependencies():
        sys.exit(1)
    print()

    # 创建目录
    print("📁 创建必要目录...")
    create_directories()
    print()

    # 启动应用
    print("🌐 启动Gradio应用...")
    print("📍 本地访问: http://localhost:7860")
    print("⚠️  按Ctrl+C停止服务")
    print("=" * 60)
    print()

    try:
        from gradio_app import demo

        demo.launch(
            server_name="0.0.0.0", server_port=7860, share=False, show_error=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
