#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils - 工具函数模块
包含各种辅助函数
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional


def scan_directory_for_images(directory_path: str, 
                              supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.tif')) -> List[Path]:
    """扫描目录中的所有图片文件"""
    image_files = []
    path = Path(directory_path)
    
    if not path.exists() or not path.is_dir():
        return []
    
    for fmt in supported_formats:
        image_files.extend(path.rglob(f'*{fmt}'))
        image_files.extend(path.rglob(f'*{fmt.upper()}'))
    
    return sorted(list(set(image_files)))


def create_detection_summary(results_list: List[dict]) -> str:
    """创建批量检测汇总报告"""
    if not results_list:
        return "暂无检测结果"
    
    total_images = len(results_list)
    total_objects = sum(r.get('object_count', 0) for r in results_list)
    avg_time = np.mean([r.get('inference_time', 0) for r in results_list])
    
    lines = []
    lines.append("=" * 60)
    lines.append("📊 批量检测汇总报告")
    lines.append("=" * 60)
    lines.append(f"📅 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"📁 处理图片数: {total_images}")
    lines.append(f"🎯 总检测目标: {total_objects}")
    lines.append(f"⏱️ 平均推理时间: {avg_time:.3f} 秒")
    lines.append("")
    
    # 统计所有类别
    all_class_counts = {}
    for result in results_list:
        for class_name, count in result.get('class_counts', {}).items():
            all_class_counts[class_name] = all_class_counts.get(class_name, 0) + count
    
    if all_class_counts:
        lines.append("📈 类别统计:")
        for class_name, count in sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"   • {class_name}: {count} 个")
        lines.append("")
    
    lines.append("📋 详细结果:")
    lines.append("-" * 60)
    
    for i, result in enumerate(results_list, 1):
        filename = Path(result.get('file_path', 'unknown')).name
        obj_count = result.get('object_count', 0)
        inf_time = result.get('inference_time', 0)
        
        lines.append(f"{i}. {filename}")
        lines.append(f"   🎯 目标数: {obj_count} | ⏱️ 耗时: {inf_time:.3f}s")
        
        if result.get('class_counts'):
            class_summary = ", ".join([f"{k}:{v}" for k, v in result['class_counts'].items()])
            lines.append(f"   📊 {class_summary}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_detection_results(results_list: List[dict], save_dir: str) -> str:
    """保存检测结果到目录"""
    save_path = Path(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = save_path / f"detection_results_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存检测报告
    report_path = result_dir / "detection_report.txt"
    report_content = create_detection_summary(results_list)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 保存结果图片
    for i, result in enumerate(results_list):
        if 'result_image' in result and result['result_image'] is not None:
            file_name = Path(result.get('file_path', f'image_{i}')).stem
            result_img = result['result_image']
            
            # 确保是BGR格式用于保存
            if len(result_img.shape) == 3 and result_img.shape[2] == 3:
                result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            else:
                result_img_bgr = result_img
            
            result_save_path = result_dir / f"{file_name}_result.jpg"
            cv2.imwrite(str(result_save_path), result_img_bgr)
    
    return str(result_dir)


def list_available_cameras(max_check: int = 4) -> List[dict]:
    """列出所有可用摄像头"""
    cameras = []
    
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                cameras.append({
                    'id': i,
                    'name': f"摄像头 {i}",
                    'resolution': f"{width}x{height}"
                })
            cap.release()
    
    return cameras


def check_camera_available(camera_id: int = 0) -> bool:
    """检查指定摄像头是否可用"""
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    return False