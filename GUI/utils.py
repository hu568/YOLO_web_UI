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


def scan_directory_for_images(
    directory_path: str,
    supported_formats: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".webp",
        ".tif",
    ),
) -> List[Path]:
    """扫描目录中的所有图片文件"""
    image_files = []
    path = Path(directory_path)

    if not path.exists() or not path.is_dir():
        return []

    for fmt in supported_formats:
        image_files.extend(path.rglob(f"*{fmt}"))
        image_files.extend(path.rglob(f"*{fmt.upper()}"))

    return sorted(list(set(image_files)))


def create_detection_summary(results_list: List[dict]) -> str:
    """创建批量检测汇总报告"""
    if not results_list:
        return "暂无检测结果"

    total_images = len(results_list)
    total_objects = sum(r.get("object_count", 0) for r in results_list)
    avg_time = np.mean([r.get("inference_time", 0) for r in results_list])

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
        for class_name, count in result.get("class_counts", {}).items():
            all_class_counts[class_name] = all_class_counts.get(class_name, 0) + count

    if all_class_counts:
        lines.append("📈 类别统计:")
        for class_name, count in sorted(
            all_class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"   • {class_name}: {count} 个")
        lines.append("")

    lines.append("📋 详细结果:")
    lines.append("-" * 60)

    for i, result in enumerate(results_list, 1):
        filename = Path(result.get("file_path", "unknown")).name
        obj_count = result.get("object_count", 0)
        inf_time = result.get("inference_time", 0)

        lines.append(f"{i}. {filename}")
        lines.append(f"   🎯 目标数: {obj_count} | ⏱️ 耗时: {inf_time:.3f}s")

        if result.get("class_counts"):
            class_summary = ", ".join(
                [f"{k}:{v}" for k, v in result["class_counts"].items()]
            )
            lines.append(f"   📊 {class_summary}")

    lines.append("=" * 60)

    return "\n".join(lines)


def save_detection_results(results_list: List[dict], save_dir: str) -> str:
    """保存检测结果到目录，包括TXT、CSV和HTML报告"""
    save_path = Path(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = save_path / f"detection_results_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存TXT检测报告
    report_path = result_dir / "detection_report.txt"
    report_content = create_detection_summary(results_list)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    # 保存CSV报告
    create_detection_csv(results_list, result_dir)

    # 保存HTML报告（带超链接）
    create_detection_html(results_list, result_dir)

    # 保存结果图片
    for i, result in enumerate(results_list):
        if "result_image" in result and result["result_image"] is not None:
            file_name = Path(result.get("file_path", f"image_{i}")).stem
            result_img = result["result_image"]

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

                cameras.append(
                    {"id": i, "name": f"摄像头 {i}", "resolution": f"{width}x{height}"}
                )
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


def create_detection_csv(results_list: List[dict], save_path: str) -> str:
    """创建CSV格式的检测报告"""
    import csv

    csv_path = Path(save_path) / "detection_report.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(
            ["序号", "文件名", "原始路径", "检测目标数", "推理时间(秒)", "类别统计"]
        )

        # 写入数据
        for i, result in enumerate(results_list, 1):
            file_path = result.get("file_path", "unknown")
            filename = Path(file_path).name
            obj_count = result.get("object_count", 0)
            inf_time = result.get("inference_time", 0)

            # 类别统计格式化为字符串
            class_counts = result.get("class_counts", {})
            class_summary = (
                "; ".join([f"{k}:{v}" for k, v in class_counts.items()])
                if class_counts
                else ""
            )

            writer.writerow(
                [i, filename, file_path, obj_count, f"{inf_time:.3f}", class_summary]
            )

    return str(csv_path)


def create_detection_html(results_list: List[dict], save_path: str) -> str:
    """创建带超链接和类别筛选功能的HTML检测报告"""
    from html import escape

    html_path = Path(save_path) / "detection_report.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 统计信息
    total_images = len(results_list)
    total_objects = sum(r.get("object_count", 0) for r in results_list)
    avg_time = (
        np.mean([r.get("inference_time", 0) for r in results_list])
        if results_list
        else 0
    )

    # 类别统计
    all_class_counts = {}
    for result in results_list:
        for class_name, count in result.get("class_counts", {}).items():
            all_class_counts[class_name] = all_class_counts.get(class_name, 0) + count

    # 收集所有唯一的类别名称用于筛选
    all_class_names = sorted(all_class_counts.keys())

    # 为每个结果生成数据属性（用于JavaScript筛选）
    results_data = []
    for i, result in enumerate(results_list, 1):
        file_path = result.get("file_path", "unknown")
        filename = Path(file_path).name
        obj_count = result.get("object_count", 0)
        inf_time = result.get("inference_time", 0)
        class_counts = result.get("class_counts", {})
        class_list = list(class_counts.keys())
        result_img_name = f"{Path(filename).stem}_result.jpg"

        results_data.append(
            {
                "index": i,
                "filename": filename,
                "result_img": result_img_name,
                "obj_count": obj_count,
                "inf_time": inf_time,
                "class_counts": class_counts,
                "class_list": class_list,
            }
        )

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO检测报告显示</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary h2 {{
            margin-top: 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-item {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        tr.hidden {{
            display: none;
        }}
        .file-link {{
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }}
        .file-link:hover {{
            text-decoration: underline;
            color: #2980b9;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge-success {{
            background-color: #27ae60;
            color: white;
        }}
        .badge-warning {{
            background-color: #f39c12;
            color: white;
        }}
        .class-tag {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 2px 8px;
            border-radius: 4px;
            margin: 2px;
            font-size: 0.85em;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .class-tag:hover {{
            background-color: #3498db;
            color: white;
        }}
        .class-tag.active {{
            background-color: #27ae60;
            color: white;
        }}
        .filter-section {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
        .filter-section h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .filter-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        .filter-btn {{
            padding: 6px 12px;
            border: 1px solid #3498db;
            background-color: white;
            color: #3498db;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }}
        .filter-btn:hover {{
            background-color: #3498db;
            color: white;
        }}
        .filter-btn.active {{
            background-color: #27ae60;
            border-color: #27ae60;
            color: white;
        }}
        .filter-btn.clear {{
            border-color: #e74c3c;
            color: #e74c3c;
        }}
        .filter-btn.clear:hover {{
            background-color: #e74c3c;
            color: white;
        }}
        .filter-info {{
            margin-top: 10px;
            padding: 8px;
            background-color: #e8f4f8;
            border-radius: 4px;
            color: #2c3e50;
            font-size: 0.9em;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 YOLO目标检测报告</h1>
        
        <div class="summary">
            <h2>检测汇总</h2>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="total-images">{total_images}</div>
                    <div class="stat-label">处理图片数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_objects}</div>
                    <div class="stat-label">检测目标总数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{avg_time:.3f}s</div>
                    <div class="stat-label">平均推理时间</div>
                </div>
            </div>
        </div>
"""

    # 添加类别统计和筛选按钮
    if all_class_names:
        html_content += """
        <div class="filter-section">
            <h3>🔍 类别筛选</h3>
            <p style="margin: 5px 0; color: #666;">点击类别按钮筛选显示包含该类别的图片：</p>
            <div class="filter-buttons">
                <button class="filter-btn clear" onclick="clearFilter()">显示全部</button>
"""
        for class_name in all_class_names:
            count = all_class_counts[class_name]
            html_content += f'                <button class="filter-btn" onclick="filterByClass(\'{escape(class_name)}\')">{escape(class_name)} ({count})</button>\n'

        html_content += """            </div>
            <div class="filter-info" id="filter-info">当前显示：全部图片</div>
        </div>
        
        <h2>📈 类别统计</h2>
        <div style="margin: 15px 0;">
"""
        for class_name, count in sorted(
            all_class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            html_content += f'            <span class="class-tag" onclick="filterByClass(\'{escape(class_name)}\')">{escape(class_name)}: {count}</span>\n'
        html_content += """        </div>
"""

    # 添加详细结果表格
    html_content += """
        <h2>📋 详细检测结果</h2>
        <table>
            <thead>
                <tr>
                    <th>序号</th>
                    <th>文件名</th>
                    <th>检测目标数</th>
                    <th>推理时间</th>
                    <th>类别统计</th>
                </tr>
            </thead>
            <tbody id="results-table">
"""

    for data in results_data:
        class_counts = data["class_counts"]
        class_list_str = ",".join(data["class_list"])

        # 类别统计
        class_summary = (
            "<br>".join([f"{escape(k)}: {v}" for k, v in class_counts.items()])
            if class_counts
            else "-"
        )

        # 目标数标签
        count_badge = (
            f'<span class="badge badge-success">{data["obj_count"]}</span>'
            if data["obj_count"] > 0
            else f'<span class="badge badge-warning">{data["obj_count"]}</span>'
        )

        html_content += f"""                <tr data-classes="{escape(class_list_str)}">
                    <td>{data["index"]}</td>
                    <td>
                        <a href="{data["result_img"]}" class="file-link" target="_blank">{escape(data["filename"])}</a>
                    </td>
                    <td>{count_badge}</td>
                    <td>{data["inf_time"]:.3f}s</td>
                    <td>{class_summary}</td>
                </tr>
"""

    html_content += f"""            </tbody>
        </table>
        
        <div class="timestamp">
            <p>📅 生成时间: {timestamp}</p>
            <p>💡 提示：点击文件名可以查看检测结果图片，点击类别标签可以筛选图片</p>
        </div>
    </div>

    <script>
        // 筛选功能
        function filterByClass(className) {{
            const rows = document.querySelectorAll('#results-table tr');
            const buttons = document.querySelectorAll('.filter-btn');
            const classTags = document.querySelectorAll('.class-tag');
            let visibleCount = 0;
            
            // 更新按钮状态
            buttons.forEach(btn => {{
                btn.classList.remove('active');
                if (btn.textContent.includes(className)) {{
                    btn.classList.add('active');
                }}
            }});
            
            // 更新标签状态
            classTags.forEach(tag => {{
                tag.classList.remove('active');
                if (tag.textContent.includes(className)) {{
                    tag.classList.add('active');
                }}
            }});
            
            // 筛选表格行
            rows.forEach(row => {{
                const classes = row.getAttribute('data-classes');
                if (classes && classes.includes(className)) {{
                    row.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    row.classList.add('hidden');
                }}
            }});
            
            // 更新筛选信息
            document.getElementById('filter-info').textContent = 
                `当前显示：包含 "${{className}}" 的图片 (${{visibleCount}} 张)`;
            document.getElementById('total-images').textContent = visibleCount;
        }}
        
        // 清除筛选
        function clearFilter() {{
            const rows = document.querySelectorAll('#results-table tr');
            const buttons = document.querySelectorAll('.filter-btn');
            const classTags = document.querySelectorAll('.class-tag');
            
            // 清除按钮状态
            buttons.forEach(btn => btn.classList.remove('active'));
            classTags.forEach(tag => tag.classList.remove('active'));
            
            // 显示所有行
            rows.forEach(row => row.classList.remove('hidden'));
            
            // 更新筛选信息
            document.getElementById('filter-info').textContent = '当前显示：全部图片';
            document.getElementById('total-images').textContent = '{total_images}';
        }}
    </script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return str(html_path)
