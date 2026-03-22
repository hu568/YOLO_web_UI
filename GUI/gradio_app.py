#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio YOLO Detection App - Gradio主应用
使用Gradio Blocks构建的YOLO目标检测Web界面
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
    check_camera_available
)

# 全局变量
model_manager = ModelManager()
current_model = None
detection_engine = None

# CSS样式
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
"""


def refresh_model_list():
    """刷新模型列表"""
    models = model_manager.get_model_list()
    if not models:
        return gr.Dropdown(choices=["无可用模型"], value="无可用模型")
    return gr.Dropdown(choices=models, value=models[0] if models else None)


def load_model(model_name: str):
    """加载模型"""
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
    """检测单张图片"""
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
    """检测视频文件"""
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
        output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 处理视频
        frame_count = 0
        for frame_count, total_frames, result_frame, inf_time, obj_count in detection_engine.process_video(
            video_path, conf_threshold
        ):
            # 写入帧
            if result_frame is not None:
                # RGB to BGR for OpenCV
                result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                out.write(result_frame_bgr)
            
            # 更新进度
            progress_percent = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            progress(progress_percent / 100)
        
        out.release()
        
        info_text = f"✅ 视频处理完成\n📊 总帧数: {frame_count}\n📁 保存路径: {output_path}"
        return output_path, info_text
        
    except Exception as e:
        return None, f"❌ 视频处理失败: {str(e)}"


def process_webcam_frame(frame, conf_threshold, model_name):
    """处理摄像头帧（用于实时流）"""
    global current_model, detection_engine
    
    if frame is None:
        return None
    
    # 确保模型已加载
    if current_model is None or detection_engine is None:
        if model_name and model_name != "无可用模型":
            load_model(model_name)
        else:
            return frame
    
    try:
        result_frame = detection_engine.process_webcam_frame(frame, conf_threshold)
        return result_frame if result_frame is not None else frame
    except Exception as e:
        print(f"摄像头处理错误: {e}")
        return frame


def batch_detect(folder_path, conf_threshold, model_name, progress=gr.Progress()):
    """批量检测文件夹"""
    if not folder_path or not Path(folder_path).exists():
        return "⚠️ 请选择有效的文件夹", None, ""
    
    # 确保模型已加载
    if current_model is None or detection_engine is None:
        load_status = load_model(model_name)
        if current_model is None:
            return f"❌ 模型未加载\n{load_status}", None, ""
    
    try:
        # 扫描图片
        image_files = scan_directory_for_images(folder_path)
        if not image_files:
            return "⚠️ 文件夹中没有找到支持的图片格式", None, ""
        
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
            results_list.append({
                'file_path': str(img_path),
                'object_count': result.object_count,
                'inference_time': result.inference_time,
                'class_counts': result.class_counts,
                'result_image': result.result_image
            })
            
            # 更新进度
            progress((i + 1) / total_files)
        
        # 生成报告
        summary = create_detection_summary(results_list)
        
        # 保存结果
        save_dir = str(Path(folder_path) / "detection_output")
        save_path = save_detection_results(results_list, save_dir)
        
        return summary, save_path, f"✅ 已保存到: {save_path}"
        
    except Exception as e:
        return f"❌ 批量检测失败: {str(e)}", None, ""


def get_camera_list():
    """获取摄像头列表"""
    cameras = list_available_cameras()
    if not cameras:
        return gr.Dropdown(choices=["未检测到摄像头"], value="未检测到摄像头")
    
    camera_options = [f"{cam['name']} ({cam['resolution']})" for cam in cameras]
    return gr.Dropdown(choices=camera_options, value=camera_options[0] if camera_options else None)


# 创建Gradio界面
with gr.Blocks(title="YOLO目标检测系统") as demo:
    
    # 标题
    gr.Markdown("""
    <div class="main-title">🚀 YOLO 目标检测系统</div>
    <div class="subtitle">基于Gradio的Web界面 | 支持图片/视频/摄像头检测</div>
    """)
    
    # 全局设置
    with gr.Row():
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                label="🤖 选择模型",
                choices=["点击刷新按钮加载模型列表"],
                value="点击刷新按钮加载模型列表",
                interactive=True
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 刷新模型列表", variant="secondary")
        with gr.Column(scale=2):
            conf_slider = gr.Slider(
                minimum=0.01,
                maximum=1.0,
                value=0.25,
                step=0.01,
                label="🎚️ 置信度阈值"
            )
    
    with gr.Row():
        load_model_btn = gr.Button("📥 加载模型", variant="primary")
        model_status = gr.Textbox(
            label="模型状态",
            value="请先加载模型",
            interactive=False
        )
    
    # 标签页
    with gr.Tabs():
        
        # 标签页1: 单张图片检测
        with gr.Tab("📷 图片检测"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="上传图片",
                        type="numpy"
                    )
                    detect_image_btn = gr.Button("🔍 开始检测", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(
                        label="检测结果",
                        type="numpy"
                    )
                    image_info = gr.Textbox(
                        label="检测信息",
                        lines=10,
                        interactive=False
                    )
            
            detect_image_btn.click(
                fn=detect_single_image,
                inputs=[input_image, conf_slider, model_dropdown],
                outputs=[output_image, image_info]
            )
        
        # 标签页2: 视频检测
        with gr.Tab("🎬 视频检测"):
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(
                        label="上传视频"
                    )
                    detect_video_btn = gr.Button("🎥 开始处理", variant="primary")
                
                with gr.Column():
                    output_video = gr.Video(
                        label="检测结果视频"
                    )
                    video_info = gr.Textbox(
                        label="处理信息",
                        lines=5,
                        interactive=False
                    )
            
            detect_video_btn.click(
                fn=detect_video,
                inputs=[input_video, conf_slider, model_dropdown],
                outputs=[output_video, video_info]
            )
        
        # 标签页3: 摄像头实时检测
        with gr.Tab("📹 实时检测"):
            with gr.Row():
                gr.Markdown("### 📹 摄像头实时检测\n点击开始后，允许浏览器访问摄像头")
            
            with gr.Row():
                with gr.Column():
                    camera_input = gr.Image(
                        label="摄像头输入",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy"
                    )
                
                with gr.Column():
                    camera_output = gr.Image(
                        label="检测结果",
                        streaming=True,
                        type="numpy"
                    )
            
            # 实时流处理
            camera_input.stream(
                fn=process_webcam_frame,
                inputs=[camera_input, conf_slider, model_dropdown],
                outputs=[camera_output],
                time_limit=300,  # 5分钟
                stream_every=0.1
            )
        
        # 标签页4: 批量检测
        with gr.Tab("📂 批量检测"):
            with gr.Row():
                with gr.Column():
                    folder_input = gr.Textbox(
                        label="📁 文件夹路径",
                        placeholder="输入包含图片的文件夹路径..."
                    )
                    batch_detect_btn = gr.Button("🚀 开始批量检测", variant="primary")
                
                with gr.Column():
                    batch_summary = gr.Textbox(
                        label="📊 检测汇总",
                        lines=15,
                        interactive=False
                    )
            
            with gr.Row():
                save_path_output = gr.Textbox(
                    label="📁 保存路径",
                    interactive=False
                )
                save_status = gr.Textbox(
                    label="保存状态",
                    interactive=False
                )
            
            batch_detect_btn.click(
                fn=batch_detect,
                inputs=[folder_input, conf_slider, model_dropdown],
                outputs=[batch_summary, save_path_output, save_status]
            )
        
        # 标签页5: 模型信息
        with gr.Tab("ℹ️ 模型信息"):
            with gr.Row():
                model_info_text = gr.Textbox(
                    label="📋 已加载模型信息",
                    lines=10,
                    interactive=False
                )
            
            def show_model_info():
                if current_model is None:
                    return "❌ 尚未加载模型\n\n请先选择一个模型并点击'加载模型'按钮"
                
                info = f"✅ 当前模型: {model_manager.current_model_path}\n\n"
                info += f"📊 类别数量: {len(model_manager.class_names)}\n\n"
                info += f"📋 类别列表:\n"
                for i, name in enumerate(model_manager.class_names, 1):
                    info += f"  {i}. {name}\n"
                
                return info
            
            refresh_info_btn = gr.Button("🔄 刷新信息")
            refresh_info_btn.click(fn=show_model_info, outputs=[model_info_text])
    
    # 事件绑定
    refresh_btn.click(fn=refresh_model_list, outputs=[model_dropdown])
    load_model_btn.click(fn=load_model, inputs=[model_dropdown], outputs=[model_status])


if __name__ == "__main__":
    # 启动应用
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
