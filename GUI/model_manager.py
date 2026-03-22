#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Manager - 模型管理模块
负责YOLO模型的扫描、加载和管理
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class ModelManager:
    """模型管理器 - 处理模型扫描和加载"""

    def __init__(self):
        # 默认模型搜索路径
        self.models_paths = [
            Path("pt_models"),
            Path("models"),
            Path("weights"),
            Path("yolo_models"),
            Path.home() / "yolo_models",
            Path("C:/yolo_models"),  # Windows系统路径
            Path("/usr/local/share/yolo_models"),  # Linux系统路径
        ]
        self.current_model = None
        self.current_model_path = None
        self.class_names = []

    def scan_models(self, custom_path: str = None) -> List[Dict]:
        """扫描模型文件"""
        models = []
        search_paths = self.models_paths.copy()

        if custom_path and Path(custom_path).exists():
            search_paths.insert(0, Path(custom_path))

        for model_dir in search_paths:
            if model_dir.exists():
                try:
                    pt_files = sorted(model_dir.glob("*.pt"))
                    for pt_file in pt_files:
                        models.append({
                            'name': pt_file.name,
                            'path': str(pt_file),
                            'size': self._get_file_size(pt_file),
                            'modified': self._get_modification_time(pt_file)
                        })
                except Exception as e:
                    print(f"扫描目录 {model_dir} 时出错: {e}")

        return models

    def get_model_list(self, custom_path: str = None) -> List[str]:
        """获取模型名称列表"""
        models = self.scan_models(custom_path)
        return [model['name'] for model in models]

    def get_model_path(self, model_name: str, custom_path: str = None) -> Optional[str]:
        """根据模型名称获取完整路径"""
        models = self.scan_models(custom_path)
        for model in models:
            if model['name'] == model_name:
                return model['path']
        return None

    def load_model(self, model_path: str):
        """加载模型"""
        try:
            from ultralytics import YOLO
            self.current_model = YOLO(model_path)
            self.current_model_path = model_path
            self.class_names = list(self.current_model.names.values())
            return self.current_model
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

    def get_current_model(self):
        """获取当前加载的模型"""
        return self.current_model

    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.class_names

    def _get_file_size(self, file_path: Path) -> str:
        """获取文件大小"""
        try:
            size = file_path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "Unknown"

    def _get_modification_time(self, file_path: Path) -> str:
        """获取文件修改时间"""
        try:
            timestamp = file_path.stat().st_mtime
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
        except:
            return "Unknown"


# 全局模型管理器实例
model_manager = ModelManager()
