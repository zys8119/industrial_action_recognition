#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时图像标注工具 - 用于标注物体位置和动作
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class ImageAnnotator:
    def __init__(self, output_dir="data/annotations"):
        """初始化标注工具"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前标注状态
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.boxes = []  # [(x1, y1, x2, y2, label), ...]
        
        # 当前帧和标注
        self.current_frame = None
        self.frame_count = 0
        
        # 类别标签
        self.labels = self.load_labels()
        self.current_label_idx = 0
        
        # 标注历史
        self.annotations = []
        
        print("🎨 图像标注工具已启动")
        print(f"📁 标注保存至: {self.output_dir}")
        self.print_help()
    
    def load_labels(self):
        """加载类别标签"""
        label_file = Path(__file__).parent.parent / "configs" / "label_list.txt"
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]
            return labels
        return ["object", "action1", "action2"]
    
    def print_help(self):
        """打印帮助信息"""
        print("\n⌨️  快捷键:")
        print("   鼠标左键拖动 - 绘制边界框")
        print("   数字键 1-9  - 切换类别标签")
        print("   SPACE       - 保存当前帧标注")
        print("   c           - 清除当前帧所有标注")
        print("   u           - 撤销最后一个标注")
        print("   s           - 导出所有标注到JSON")
        print("   h           - 显示帮助")
        print("   q           - 退出")
        print(f"\n📋 当前类别: {self.labels}")
        print()
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_point[0], self.start_point[1], x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # 确保坐标正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 只保存有效的框（面积 > 100）
                if (x2 - x1) * (y2 - y1) > 100:
                    label = self.labels[self.current_label_idx]
                    self.boxes.append((x1, y1, x2, y2, label))
                    print(f"✅ 添加标注: {label} at ({x1},{y1})-({x2},{y2})")
                
                self.current_box = None
                self.start_point = None
    
    def draw_annotations(self, frame):
        """在画面上绘制标注"""
        display = frame.copy()
        
        # 绘制已保存的框
        for x1, y1, x2, y2, label in self.boxes:
            color = self.get_label_color(label)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label_text = label
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(display, label_text, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制正在绘制的框
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            label = self.labels[self.current_label_idx]
            color = self.get_label_color(label)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制状态栏
        self.draw_status_bar(display)
        
        return display
    
    def get_label_color(self, label):
        """根据标签获取颜色"""
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
        ]
        idx = self.labels.index(label) if label in self.labels else 0
        return colors[idx % len(colors)]
    
    def draw_status_bar(self, frame):
        """绘制状态栏"""
        h, w = frame.shape[:2]
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 当前类别
        current_label = self.labels[self.current_label_idx]
        color = self.get_label_color(current_label)
        cv2.putText(frame, f"Current: {current_label} (Press 1-{len(self.labels)})",
                   (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 标注数量
        cv2.putText(frame, f"Boxes: {len(self.boxes)} | Frames: {self.frame_count}",
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 快捷键提示
        cv2.putText(frame, "SPACE:Save | C:Clear | U:Undo | S:Export | H:Help | Q:Quit",
                   (w - 650, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def save_current_frame(self):
        """保存当前帧和标注"""
        if self.current_frame is None or len(self.boxes) == 0:
            print("⚠️  没有标注可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 保存图像
        img_filename = f"frame_{timestamp}.jpg"
        img_path = self.output_dir / img_filename
        cv2.imwrite(str(img_path), self.current_frame)
        
        # 保存标注
        annotation = {
            "image": img_filename,
            "timestamp": timestamp,
            "width": self.current_frame.shape[1],
            "height": self.current_frame.shape[0],
            "boxes": [
                {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": label,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
                for x1, y1, x2, y2, label in self.boxes
            ]
        }
        
        self.annotations.append(annotation)
        self.frame_count += 1
        
        print(f"💾 已保存: {img_filename} ({len(self.boxes)} 个标注)")
        
        # 清除当前标注，准备下一帧
        self.boxes = []
    
    def export_annotations(self):
        """导出所有标注到JSON"""
        if not self.annotations:
            print("⚠️  没有标注可导出")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"annotations_{timestamp}.json"
        json_path = self.output_dir / json_filename
        
        export_data = {
            "version": "1.0",
            "created_at": timestamp,
            "labels": self.labels,
            "total_frames": len(self.annotations),
            "total_boxes": sum(len(ann["boxes"]) for ann in self.annotations),
            "annotations": self.annotations
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📦 标注已导出: {json_path}")
        print(f"   总帧数: {len(self.annotations)}")
        print(f"   总标注: {export_data['total_boxes']}")
    
    def run(self, camera_id=0):
        """运行标注工具"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
        
        window_name = "Image Annotator"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("✅ 摄像头已打开，开始标注...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            self.current_frame = frame.copy()
            
            # 绘制标注
            display = self.draw_annotations(frame)
            
            # 显示
            cv2.imshow(window_name, display)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' '):  # SPACE - 保存
                self.save_current_frame()
            
            elif key == ord('c'):  # 清除
                self.boxes = []
                print("🗑️  已清除当前标注")
            
            elif key == ord('u'):  # 撤销
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"↩️  已撤销: {removed[4]}")
            
            elif key == ord('s'):  # 导出
                self.export_annotations()
            
            elif key == ord('h'):  # 帮助
                self.print_help()
            
            elif ord('1') <= key <= ord('9'):  # 切换类别
                idx = key - ord('1')
                if idx < len(self.labels):
                    self.current_label_idx = idx
                    print(f"🏷️  切换到类别: {self.labels[idx]}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 退出时自动导出
        if self.annotations:
            self.export_annotations()
        
        print("👋 标注工具已退出")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='实时图像标注工具')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--output', type=str, default='data/annotations', help='输出目录')
    args = parser.parse_args()
    
    annotator = ImageAnnotator(args.output)
    annotator.run(args.camera)

if __name__ == "__main__":
    main()
