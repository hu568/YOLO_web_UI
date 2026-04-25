#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Script - 启动脚本
启动Gradio YOLO检测应用

Gradio 6.9.0 兼容版本
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """检查必要的依赖

    Returns:
        bool: 所有依赖是否都已安装
    """
    missing = []

    try:
        import gradio

        print(f"[OK] Gradio 版本: {gradio.__version__}")
        # 检查版本是否为6.9.0
        if gradio.__version__ != "6.9.0":
            print(f"[!] 警告: 当前Gradio版本为 {gradio.__version__}，推荐使用 6.9.0")
    except ImportError:
        missing.append("gradio==6.9.0")

    try:
        import ultralytics

        print(f"[OK] Ultralytics 已安装")
    except ImportError:
        missing.append("ultralytics")

    try:
        import cv2

        print(f"[OK] OpenCV 版本: {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy

        print(f"[OK] NumPy 版本: {numpy.__version__}")
    except ImportError:
        missing.append("numpy")

    try:
        from PIL import Image

        print(f"[OK] Pillow 已安装")
    except ImportError:
        missing.append("pillow")

    try:
        from fastrtc import WebRTC

        print(f"[OK] FastRTC 已安装（实时检测可用）")
    except ImportError:
        print(f"[!] FastRTC 未安装（实时检测功能将不可用）")
        print(f"   安装命令: pip install fastrtc")

    if missing:
        print("\n[X] 缺少以下依赖包:")
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

    print("[OK] 目录检查完成")


def main():
    """主函数"""
    print("=" * 60)
    print("YOLO目标检测系统 - Gradio 6.9.0版本")
    print("=" * 60)
    print()

    # 检查依赖
    print("[检查依赖...]")
    if not check_dependencies():
        sys.exit(1)
    print()

    # 创建目录
    print("[创建必要目录...]")
    create_directories()
    print()

    # 启动应用
    print("[启动Gradio应用...]")
    print("本地访问: http://localhost:7860")
    print("[!] 按Ctrl+C停止服务")
    print("=" * 60)
    print()

    try:
        from gradio_app import demo, custom_css

        # Gradio 5.x/6.x: css 参数在 Blocks() 中传入
        demo.launch(
            server_name="0.0.0.0", server_port=7860, share=False, show_error=True
        )
    except KeyboardInterrupt:
        print("\n\n[服务已停止]")
    except Exception as e:
        print(f"\n[X] 启动失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
