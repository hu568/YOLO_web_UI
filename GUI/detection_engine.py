#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection Engine - 检测引擎模块
负责执行YOLO目标检测的核心逻辑
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Generator
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """检测结果数据类"""

    original_image: np.ndarray
    result_image: np.ndarray
    inference_time: float
    object_count: int
    class_counts: Dict[str, int]
    detections: List[Dict]
    class_names: List[str]


class DetectionEngine:
    """检测引擎 - 处理各种检测任务"""

    def __init__(self, model):
        self.model = model
        self.class_names = list(model.names.values()) if model else []

    def detect_image(
        self, image: np.ndarray, conf_threshold: float = 0.25
    ) -> DetectionResult:
        """检测单张图片"""
        if image is None:
            raise ValueError("图片为空")

        # 确保图片格式正确
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 执行检测
        start_time = time.time()
        results = self.model(image, conf=conf_threshold, verbose=False)
        end_time = time.time()

        # 处理结果
        inference_time = end_time - start_time
        result_image = results[0].plot()

        # YOLO的plot()方法返回的是BGR格式，需要转换为RGB用于显示
        # 注意：不要重复转换，否则会导致颜色反转（负片效果）
        if len(result_image.shape) == 3 and result_image.shape[2] == 3:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # 提取检测信息
        detections = []
        class_counts = {}
        object_count = 0

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()

            object_count = len(confidences)

            for i, (conf, cls, box) in enumerate(zip(confidences, classes, xyxy)):
                class_name = (
                    self.class_names[cls]
                    if cls < len(self.class_names)
                    else f"类别{cls}"
                )

                detections.append(
                    {
                        "id": i + 1,
                        "class": class_name,
                        "confidence": float(conf),
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ],
                        "width": float(box[2] - box[0]),
                        "height": float(box[3] - box[1]),
                    }
                )

                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return DetectionResult(
            original_image=image,
            result_image=result_image,
            inference_time=inference_time,
            object_count=object_count,
            class_counts=class_counts,
            detections=detections,
            class_names=self.class_names,
        )

    def process_video(
        self, video_path: str, conf_threshold: float = 0.25, progress_callback=None
    ):
        """处理视频文件（生成器）"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 转换帧为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 检测
                start_time = time.time()
                results = self.model(frame_rgb, conf=conf_threshold, verbose=False)
                end_time = time.time()

                result_frame = results[0].plot()

                # YOLO的plot()返回BGR格式，转换为RGB
                if len(result_frame.shape) == 3 and result_frame.shape[2] == 3:
                    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

                object_count = 0
                if results[0].boxes is not None:
                    object_count = len(results[0].boxes)

                frame_count += 1

                if progress_callback:
                    progress = (
                        int((frame_count / total_frames) * 100)
                        if total_frames > 0
                        else 0
                    )
                    progress_callback(progress, frame_count, total_frames)

                yield (
                    frame_count,
                    total_frames,
                    result_frame,
                    end_time - start_time,
                    object_count,
                )

        finally:
            cap.release()

    def process_webcam_frame(
        self, frame: np.ndarray, conf_threshold: float = 0.25
    ) -> np.ndarray:
        """处理摄像头帧"""
        if frame is None:
            return None

        # 确保是RGB格式
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        results = self.model(frame_rgb, conf=conf_threshold, verbose=False)
        result_frame = results[0].plot()

        # YOLO的plot()返回BGR格式，转换为RGB用于Gradio显示
        if len(result_frame.shape) == 3 and result_frame.shape[2] == 3:
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        return result_frame


def format_detection_info(result: DetectionResult) -> str:
    """格式化检测信息"""
    info_lines = []
    info_lines.append(f"🎯 检测到 {result.object_count} 个目标")
    info_lines.append(f"⏱️ 推理耗时: {result.inference_time:.3f} 秒")

    if result.detections:
        avg_conf = np.mean([d["confidence"] for d in result.detections])
        info_lines.append(f"📊 平均置信度: {avg_conf:.3f}")

        info_lines.append("\n📋 类别统计:")
        for class_name, count in sorted(result.class_counts.items()):
            info_lines.append(f"   • {class_name}: {count} 个")

        info_lines.append("\n🔍 详细检测结果:")
        for det in result.detections[:10]:
            info_lines.append(
                f"   #{det['id']}: {det['class']} (置信度: {det['confidence']:.3f})"
            )

        if len(result.detections) > 10:
            info_lines.append(f"   ... 还有 {len(result.detections) - 10} 个目标")

    return "\n".join(info_lines)
