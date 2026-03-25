#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio YOLO Detection App - Gradio 6.9.0 主应用
使用Gradio Blocks构建的YOLO目标检测Web界面

Gradio 6.9.0 重要变更：
- theme/css/js 参数从 gr.Blocks() 移至 launch() 方法
- 统一的事件驱动架构
- 标准化的组件接口
- 原生支持 streaming 实时视频流（替代 fastrtc）
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional

import gradio as gr

# 导入自定义模块
from model_manager import ModelManager
from detection_engine import DetectionEngine, format_detection_info
from utils import (
    scan_directory_for_images,
    create_detection_summary,
    save_detection_results,
    list_available_cameras,
    check_camera_available,
)

# 全局变量
model_manager = ModelManager()
current_model = None
detection_engine = None

# CSS样式 - Gradio 6.x 中通过 launch() 方法传入
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-title {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0.5em;
}
.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 1em;
}
.info-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.result-stats {
    background: #f8f9fa;
    border-left: 4px solid #3498db;
    padding: 15px;
    margin: 10px 0;
    border-radius: 0 10px 10px 0;
}
.webcam-container {
    max-width: 640px !important;
    max-height: 480px !important;
}
"""


def refresh_model_list():
    """刷新模型列表"""
    models = model_manager.get_model_list()
    if not models:
        return gr.Dropdown(choices=["无可用模型"], value="无可用模型")
    return gr.Dropdown(choices=models, value=models[0] if models else None)


def load_model(model_name: str):
    """加载模型

    Args:
        model_name: 模型名称

    Returns:
        加载状态信息
    """
    global current_model, detection_engine

    if model_name == "无可用模型":
        return "❌ 请先扫描可用的YOLO模型"

    try:
        model_path = model_manager.get_model_path(model_name)
        if model_path and Path(model_path).exists():
            current_model = model_manager.load_model(model_path)
            detection_engine = DetectionEngine(current_model)
            return f"✅ 模型加载成功: {model_name}\n📊 类别数量: {len(model_manager.get_class_names())}"
        else:
            return f"❌ 模型文件不存在: {model_name}"
    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"


def detect_single_image(image, conf_threshold, model_name):
    """检测单张图片

    Args:
        image: 输入图像
        conf_threshold: 置信度阈值
        model_name: 模型名称

    Returns:
        (结果图像, 检测信息)
    """
    if image is None:
        return None, "⚠️ 请先上传图片"

    # 确保模型已加载
    if current_model is None or detection_engine is None:
        load_status = load_model(model_name)
        if current_model is None:
            return None, f"❌ 模型未加载\n{load_status}"

    try:
        # 执行检测
        result = detection_engine.detect_image(image, conf_threshold)

        # 格式化信息
        info_text = format_detection_info(result)

        return result.result_image, info_text
    except Exception as e:
        return None, f"❌ 检测失败: {str(e)}"


def detect_video(video_path, conf_threshold, model_name, progress=gr.Progress()):
    """检测视频文件

    Args:
        video_path: 视频文件路径
        conf_threshold: 置信度阈值
        model_name: 模型名称
        progress: Gradio进度条对象

    Returns:
        (输出视频路径, 处理信息)
    """
    if video_path is None:
        return None, "⚠️ 请先上传视频文件"

    # 确保模型已加载
    if current_model is None or detection_engine is None:
        load_status = load_model(model_name)
        if current_model is None:
            return None, f"❌ 模型未加载\n{load_status}"

    try:
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "❌ 无法打开视频文件"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 创建输出视频
        output_path = str(
            Path(video_path).parent / f"{Path(video_path).stem}_detected.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 处理视频
        frame_count = 0
        for (
            frame_count,
            total_frames,
            result_frame,
            inf_time,
            obj_count,
        ) in detection_engine.process_video(video_path, conf_threshold):
            # 写入帧
            if result_frame is not None:
                # detection_engine.process_video 现在返回RGB格式的帧
                # 需要转换为BGR格式供OpenCV VideoWriter使用
                result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                out.write(result_frame_bgr)

            # 更新进度
            progress_percent = (
                int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            )
            progress(progress_percent / 100)

        out.release()

        info_text = (
            f"✅ 视频处理完成\n📊 总帧数: {frame_count}\n📁 保存路径: {output_path}"
        )
        return output_path, info_text

    except Exception as e:
        return None, f"❌ 视频处理失败: {str(e)}"


# ==================== Gradio 6.x 原生 Streaming 实时检测 ====================


def webcam_detection_stream(frame, conf_threshold):
    """
    Gradio 6.x 原生 Streaming 实时检测处理函数
    使用 gr.Image(streaming=True) 替代 fastrtc WebRTC

    Args:
        frame: 输入视频帧 (numpy数组，RGB格式)
        conf_threshold: 置信度阈值

    Returns:
        处理后的视频帧 (RGB格式)
    """
    global current_model, detection_engine

    if frame is None:
        # 返回空白帧
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # 确保模型已加载
    if current_model is None or detection_engine is None:
        # 模型未加载时返回原始帧，并在图像上添加提示文字
        frame_with_text = frame.copy()
        cv2.putText(
            frame_with_text,
            "Please load model first",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        return frame_with_text

    try:
        # frame 已经是 RGB 格式（Gradio 自动转换）
        # 执行 YOLO 检测
        results = current_model(frame, conf=conf_threshold, verbose=False)
        result_frame = results[0].plot()

        # YOLO的plot()返回BGR格式，需要转换为RGB
        if len(result_frame.shape) == 3 and result_frame.shape[2] == 3:
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        return result_frame
    except Exception as e:
        print(f"实时检测错误: {e}")
        import traceback

        traceback.print_exc()
        # 出错时返回原始帧，并添加错误提示
        frame_with_text = frame.copy()
        cv2.putText(
            frame_with_text,
            f"Error: {str(e)[:30]}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        return frame


def batch_detect(folder_files, conf_threshold, model_name, progress=gr.Progress()):
    """批量检测上传的文件夹中的图片，并打包成ZIP供下载

    Args:
        folder_files: 文件夹中的文件列表
        conf_threshold: 置信度阈值
        model_name: 模型名称
        progress: Gradio进度条对象

    Returns:
        (汇总信息, ZIP文件路径, CSV文件路径, HTML文件路径, 保存状态)
    """
    if not folder_files or len(folder_files) == 0:
        return "⚠️ 请选择要检测的文件夹", None, None, None, ""

    # 确保模型已加载
    if current_model is None or detection_engine is None:
        load_status = load_model(model_name)
        if current_model is None:
            return f"❌ 模型未加载\n{load_status}", None, None, None, ""

    try:
        # 从上传的文件夹中收集所有图片文件
        # folder_files 是一个文件列表，包含文件夹中的所有文件
        image_files = []
        supported_formats = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".gif",
            ".tiff",
            ".tif",
        )

        for file_info in folder_files:
            # Gradio上传的文件可能是字典或字符串路径
            if isinstance(file_info, dict):
                file_path = file_info.get("name", "")
                file_path = file_info.get("path", file_path)  # 优先使用path
            elif isinstance(file_info, str):
                file_path = file_info
            else:
                # 可能是FileData对象
                file_path = getattr(file_info, "name", str(file_info))

            # 检查是否是支持的图片格式
            if file_path.lower().endswith(supported_formats):
                image_files.append(file_path)

        if not image_files:
            return (
                f"⚠️ 文件夹中没有找到支持的图片格式\n支持的格式: {', '.join(supported_formats)}",
                None,
                None,
                None,
                "",
            )

        total_files = len(image_files)
        results_list = []

        for i, img_path in enumerate(image_files):
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 检测
            result = detection_engine.detect_image(img, conf_threshold)

            # 保存结果
            results_list.append(
                {
                    "file_path": str(img_path),
                    "object_count": result.object_count,
                    "inference_time": result.inference_time,
                    "class_counts": result.class_counts,
                    "result_image": result.result_image,
                }
            )

            # 更新进度
            progress((i + 1) / total_files)

        # 生成报告
        summary = create_detection_summary(results_list)

        # 保存结果到临时目录并打包成ZIP
        import tempfile
        import zipfile
        from datetime import datetime

        # 创建临时目录保存结果
        save_dir = tempfile.mkdtemp(prefix="detection_output_")
        result_dir = save_detection_results(results_list, save_dir)

        # 创建ZIP文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"detection_results_{timestamp}.zip"
        zip_path = os.path.join(save_dir, zip_filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, save_dir)
                    zipf.write(file_path, arcname)

        # 获取CSV和HTML报告路径
        result_path = Path(result_dir)
        csv_path = str(result_path / "detection_report.csv")
        html_path = str(result_path / "detection_report.html")

        return (
            summary,
            zip_path,
            csv_path,
            html_path,
            f"✅ 检测完成！共处理 {len(results_list)} 张图片。ZIP包含所有结果，CSV和HTML报告可单独下载",
        )

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        print(f"批量检测错误: {error_detail}")
        return f"❌ 批量检测失败: {str(e)}", None, None, None, ""


def show_model_info():
    """显示模型信息

    Returns:
        模型信息文本
    """
    if current_model is None:
        return "❌ 尚未加载模型\n\n请先选择一个模型并点击'加载模型'按钮"

    info = f"✅ 当前模型: {model_manager.current_model_path}\n\n"
    info += f"📊 类别数量: {len(model_manager.class_names)}\n\n"
    info += f"📋 类别列表:\n"
    for i, name in enumerate(model_manager.class_names, 1):
        info += f"  {i}. {name}\n"

    return info


# ==================== Gradio 6.9.0 界面构建 ====================

# Gradio 6.x: theme/css 参数移至 launch() 方法
with gr.Blocks(title="YOLO目标检测系统") as demo:
    # 标题
    gr.Markdown("""
    <div class="main-title">🚀 YOLO 目标检测系统</div>
    <div class="subtitle">基于Gradio 6.9.0的Web界面 | 支持图片/视频/摄像头检测</div>
    """)

    # 全局设置
    with gr.Row():
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                label="🤖 选择模型",
                choices=["点击刷新按钮加载模型列表"],
                value="点击刷新按钮加载模型列表",
                interactive=True,
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 刷新模型列表", variant="secondary")
        with gr.Column(scale=2):
            conf_slider = gr.Slider(
                minimum=0.01, maximum=1.0, value=0.25, step=0.01, label="🎚️ 置信度阈值"
            )

    with gr.Row():
        load_model_btn = gr.Button("📥 加载模型", variant="primary")
        model_status = gr.Textbox(
            label="模型状态", value="请先加载模型", interactive=False
        )

    # 标签页
    with gr.Tabs():
        # 标签页1: 单张图片检测
        with gr.Tab("📷 图片检测"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="上传图片", type="numpy")
                    detect_image_btn = gr.Button("🔍 开始检测", variant="primary")

                with gr.Column():
                    output_image = gr.Image(label="检测结果", type="numpy")
                    image_info = gr.Textbox(
                        label="检测信息", lines=10, interactive=False
                    )

            detect_image_btn.click(
                fn=detect_single_image,
                inputs=[input_image, conf_slider, model_dropdown],
                outputs=[output_image, image_info],
            )

        # 标签页2: 视频检测
        with gr.Tab("🎬 视频检测"):
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label="上传视频")
                    detect_video_btn = gr.Button("🎥 开始处理", variant="primary")

                with gr.Column():
                    output_video = gr.Video(label="检测结果视频")
                    video_info = gr.Textbox(
                        label="处理信息", lines=5, interactive=False
                    )

            detect_video_btn.click(
                fn=detect_video,
                inputs=[input_video, conf_slider, model_dropdown],
                outputs=[output_video, video_info],
            )

        # 标签页3: 摄像头实时检测 (Gradio 6.x 原生 Streaming)
        with gr.Tab("📹 实时检测"):
            gr.Markdown("""
            ### 📹 摄像头实时目标检测
            
            **使用说明：**
            1. 先点击"📥 加载模型"按钮加载YOLO模型
            2. 然后点击下方"启动摄像头"按钮
            3. 允许浏览器访问摄像头
            4. 系统会实时显示检测结果
            
            **技术说明：** 使用 Gradio 6.x 原生 streaming 功能（替代 fastrtc）
            """)

            with gr.Row():
                with gr.Column():
                    # Gradio 6.x 原生 webcam streaming
                    webcam_input = gr.Image(
                        label="摄像头输入",
                        sources=["webcam"],
                        type="numpy",
                        streaming=True,
                        elem_classes=["webcam-container"],
                    )

                    with gr.Row():
                        start_webcam_btn = gr.Button("🎥 启动摄像头", variant="primary")
                        stop_webcam_btn = gr.Button("⏹️ 停止", variant="secondary")

                with gr.Column():
                    # 实时检测结果输出
                    webcam_output = gr.Image(
                        label="实时检测结果",
                        type="numpy",
                        streaming=True,
                        elem_classes=["webcam-container"],
                    )

                    webcam_status = gr.Textbox(
                        label="检测状态",
                        value="请先加载模型，然后启动摄像头",
                        interactive=False,
                        lines=2,
                    )

            # Gradio 6.x streaming 事件绑定
            # 使用 .stream() 方法实现实时视频流处理
            webcam_input.stream(
                fn=webcam_detection_stream,
                inputs=[webcam_input, conf_slider],
                outputs=[webcam_output],
                time_limit=300,  # 5分钟超时
                stream_every=0.1,  # 每100ms处理一帧
            )

            # 状态更新
            def update_webcam_status_start():
                if current_model is None:
                    return "❌ 模型未加载 - 请先加载模型"
                return "✅ 摄像头已启动 - 正在实时检测"

            def update_webcam_status_stop():
                return "⏹️ 摄像头已停止"

            start_webcam_btn.click(
                fn=update_webcam_status_start, outputs=[webcam_status]
            )

            stop_webcam_btn.click(fn=update_webcam_status_stop, outputs=[webcam_status])

        # 标签页4: 批量检测
        with gr.Tab("📂 批量检测"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""### 📁 上传文件夹
支持上传整个文件夹，系统会自动识别其中的图片文件（.jpg, .jpeg, .png, .bmp, .webp）""")
                    batch_folder_input = gr.File(
                        label="📁 选择文件夹",
                        file_count="directory",
                    )
                    batch_detect_btn = gr.Button("🚀 开始批量检测", variant="primary")

                with gr.Column():
                    batch_summary = gr.Textbox(
                        label="📊 检测汇总", lines=15, interactive=False
                    )

            with gr.Row():
                with gr.Column():
                    download_output = gr.File(
                        label="📥 下载完整结果 (ZIP)",
                        interactive=False,
                    )
                with gr.Column():
                    csv_download = gr.File(
                        label="📊 下载CSV报告",
                        interactive=False,
                    )
                with gr.Column():
                    html_download = gr.File(
                        label="🌐 下载HTML报告",
                        interactive=False,
                    )

            with gr.Row():
                save_status = gr.Textbox(label="保存状态", interactive=False)

            batch_detect_btn.click(
                fn=batch_detect,
                inputs=[batch_folder_input, conf_slider, model_dropdown],
                outputs=[
                    batch_summary,
                    download_output,
                    csv_download,
                    html_download,
                    save_status,
                ],
            )

        # 标签页5: 模型信息
        with gr.Tab("ℹ️ 模型信息"):
            with gr.Row():
                model_info_text = gr.Textbox(
                    label="📋 已加载模型信息", lines=10, interactive=False
                )

            refresh_info_btn = gr.Button("🔄 刷新信息")
            refresh_info_btn.click(fn=show_model_info, outputs=[model_info_text])

    # 事件绑定
    refresh_btn.click(fn=refresh_model_list, outputs=[model_dropdown])
    load_model_btn.click(fn=load_model, inputs=[model_dropdown], outputs=[model_status])


if __name__ == "__main__":
    # Gradio 6.x: theme/css/js 参数移至 launch() 方法
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=custom_css,  # Gradio 6.x 中通过 launch() 传入CSS
    )
