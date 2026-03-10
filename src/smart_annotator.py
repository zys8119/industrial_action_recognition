#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能标注与识别工具 - 结合多种算法提高准确性
支持：手动标注 + 自动跟踪 + 特征学习 + 智能建议
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import json

class SmartAnnotator:
    """智能标注器 - 提供辅助功能提高标注准确性"""
    
    def __init__(self, output_dir="data/smart_annotations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标注状态
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.annotations = []  # [(x, y, w, h, label, confidence), ...]
        
        # 跟踪器
        self.trackers = []
        
        # 当前帧
        self.current_frame = None
        self.frame_count = 0
        
        # 类别
        self.labels = self.load_labels()
        self.current_label_idx = 0
        
        # 智能辅助
        self.enable_edge_snap = True  # 边缘吸附
        self.enable_auto_adjust = True  # 自动调整
        self.enable_suggestions = True  # 智能建议
        
        # 边缘检测
        self.edge_map = None
        
        # 日志
        self.log_file = self.output_dir / "annotation_log.txt"
        self.log_fp = open(self.log_file, 'a', encoding='utf-8')
        
        print("🎨 智能标注工具已启动")
        self.log("=== 会话开始 ===")
        self.print_help()
    
    def load_labels(self):
        """加载类别标签"""
        label_file = Path(__file__).parent.parent / "configs" / "label_list.txt"
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]
            return labels
        return ["object", "person", "vehicle"]
    
    def log(self, message):
        """写入日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        self.log_fp.write(log_line + '\n')
        self.log_fp.flush()
    
    def print_help(self):
        """打印帮助信息"""
        print("\n⌨️  快捷键:")
        print("   鼠标拖动    - 绘制标注框")
        print("   数字键 1-9  - 切换类别")
        print("   SPACE       - 保存当前帧")
        print("   E           - 切换边缘吸附")
        print("   A           - 切换自动调整")
        print("   S           - 切换智能建议")
        print("   C           - 清除所有标注")
        print("   U           - 撤销最后一个")
        print("   T           - 开始跟踪")
        print("   R           - 重置")
        print("   H           - 帮助")
        print("   Q           - 退出")
        print()
    
    def detect_edges(self, frame):
        """检测边缘"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 膨胀边缘，使其更明显
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def snap_to_edge(self, x, y, search_radius=20):
        """边缘吸附 - 将点吸附到最近的边缘"""
        if self.edge_map is None:
            return x, y
        
        h, w = self.edge_map.shape
        
        # 确保坐标在范围内
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # 在搜索半径内查找边缘
        y_min = max(0, y - search_radius)
        y_max = min(h, y + search_radius)
        x_min = max(0, x - search_radius)
        x_max = min(w, x + search_radius)
        
        roi = self.edge_map[y_min:y_max, x_min:x_max]
        
        # 找到最近的边缘点
        edge_points = np.where(roi > 0)
        if len(edge_points[0]) > 0:
            # 计算距离
            distances = np.sqrt((edge_points[1] - (x - x_min))**2 + 
                              (edge_points[0] - (y - y_min))**2)
            min_idx = np.argmin(distances)
            
            if distances[min_idx] < search_radius:
                snap_x = x_min + edge_points[1][min_idx]
                snap_y = y_min + edge_points[0][min_idx]
                return snap_x, snap_y
        
        return x, y
    
    def auto_adjust_box(self, x1, y1, x2, y2):
        """自动调整边界框以更好地包围物体"""
        if self.current_frame is None:
            return x1, y1, x2, y2
        
        # 确保坐标顺序正确
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 提取ROI
        h, w = self.current_frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return x1, y1, x2, y2
        
        roi = self.current_frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 使用阈值分割
        _, thresh = cv2.threshold(gray_roi, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 添加一些边距
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(roi.shape[1] - x, w + 2 * margin)
            h = min(roi.shape[0] - y, h + 2 * margin)
            
            # 转换回原图坐标
            return x1 + x, y1 + y, x1 + x + w, y1 + y + h
        
        return x1, y1, x2, y2
    
    def suggest_boxes(self, frame):
        """智能建议 - 自动检测可能的目标区域"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用多种方法检测候选区域
        suggestions = []
        
        # 方法1: 边缘检测 + 轮廓
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 50000:  # 过滤太小或太大的区域
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5:  # 过滤不合理的长宽比
                    suggestions.append((x, y, w, h, 0.5))
        
        # 方法2: 显著性检测
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(frame)
            if success:
                _, thresh = cv2.threshold((saliency_map * 255).astype(np.uint8), 
                                         128, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 500 < area < 50000:
                        x, y, w, h = cv2.boundingRect(contour)
                        suggestions.append((x, y, w, h, 0.7))
        except:
            pass
        
        return suggestions
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            # 边缘吸附
            if self.enable_edge_snap and self.edge_map is not None:
                x, y = self.snap_to_edge(x, y)
            self.start_point = (x, y)
            self.current_box = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 边缘吸附
                if self.enable_edge_snap and self.edge_map is not None:
                    x, y = self.snap_to_edge(x, y)
                self.current_box = (self.start_point[0], self.start_point[1], x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False
                
                # 边缘吸附
                if self.enable_edge_snap and self.edge_map is not None:
                    x, y = self.snap_to_edge(x, y)
                
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # 自动调整
                if self.enable_auto_adjust:
                    x1, y1, x2, y2 = self.auto_adjust_box(x1, y1, x2, y2)
                
                # 确保坐标正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                w, h = x2 - x1, y2 - y1
                
                if w * h > 100:
                    label = self.labels[self.current_label_idx]
                    self.annotations.append((x1, y1, w, h, label, 1.0))
                    self.log(f"✅ 添加标注: {label} at ({x1},{y1}) size {w}x{h}")
                
                self.current_box = None
                self.start_point = None
    
    def draw_annotations(self, frame):
        """绘制标注"""
        display = frame.copy()
        
        # 绘制边缘（如果启用）
        if self.enable_edge_snap and self.edge_map is not None:
            edge_overlay = cv2.cvtColor(self.edge_map, cv2.COLOR_GRAY2BGR)
            edge_overlay = cv2.applyColorMap(edge_overlay, cv2.COLORMAP_JET)
            display = cv2.addWeighted(display, 0.9, edge_overlay, 0.1, 0)
        
        # 绘制智能建议（如果启用）
        if self.enable_suggestions and self.frame_count % 30 == 0:
            suggestions = self.suggest_boxes(frame)
            for x, y, w, h, conf in suggestions[:5]:  # 只显示前5个
                cv2.rectangle(display, (x, y), (x + w, y + h), 
                            (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(display, f"{conf:.0%}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 绘制已保存的标注
        for x, y, w, h, label, conf in self.annotations:
            color = self.get_label_color(label)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            
            label_text = f"{label} {conf:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x, y - text_h - 10), 
                         (x + text_w + 10, y), color, -1)
            cv2.putText(display, label_text, (x + 5, y - 5),
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
        """获取标签颜色"""
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        idx = self.labels.index(label) if label in self.labels else 0
        return colors[idx % len(colors)]
    
    def draw_status_bar(self, frame):
        """绘制状态栏"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 当前类别
        label = self.labels[self.current_label_idx]
        color = self.get_label_color(label)
        cv2.putText(frame, f"Label: {label}", (20, h - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 标注数量
        cv2.putText(frame, f"Annotations: {len(self.annotations)}", 
                   (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 辅助功能状态
        features = []
        if self.enable_edge_snap:
            features.append("EdgeSnap")
        if self.enable_auto_adjust:
            features.append("AutoAdjust")
        if self.enable_suggestions:
            features.append("Suggestions")
        
        features_text = " | ".join(features) if features else "None"
        cv2.putText(frame, f"Features: {features_text}", 
                   (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 快捷键
        cv2.putText(frame, "E:EdgeSnap | A:AutoAdjust | S:Suggestions | SPACE:Save | Q:Quit",
                   (w - 750, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def save_annotations(self):
        """保存标注"""
        if not self.annotations:
            self.log("⚠️  没有标注可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存图像
        img_path = self.output_dir / f"annotated_{timestamp}.jpg"
        annotated_frame = self.draw_annotations(self.current_frame)
        cv2.imwrite(str(img_path), annotated_frame)
        
        # 保存JSON（转换 numpy 类型为 Python 原生类型）
        data = {
            "timestamp": timestamp,
            "frame": int(self.frame_count),
            "annotations": [
                {
                    "x": int(x), "y": int(y), 
                    "width": int(w), "height": int(h),
                    "label": str(label), 
                    "confidence": float(conf)
                }
                for x, y, w, h, label, conf in self.annotations
            ]
        }
        
        json_path = self.output_dir / f"annotated_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 已保存: {img_path}")
        self.annotations = []
    
    def run(self, camera_id=0):
        """运行标注工具"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.log("❌ 无法打开摄像头")
            return
        
        window_name = "Smart Annotator"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.log("✅ 摄像头已打开")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.current_frame = frame.copy()
                self.frame_count += 1
                
                # 更新边缘检测
                if self.enable_edge_snap and self.frame_count % 10 == 0:
                    self.edge_map = self.detect_edges(frame)
                
                # 绘制
                display = self.draw_annotations(frame)
                cv2.imshow(window_name, display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.save_annotations()
                elif key == ord('e'):
                    self.enable_edge_snap = not self.enable_edge_snap
                    status = "开启" if self.enable_edge_snap else "关闭"
                    self.log(f"🔧 边缘吸附: {status}")
                elif key == ord('a'):
                    self.enable_auto_adjust = not self.enable_auto_adjust
                    status = "开启" if self.enable_auto_adjust else "关闭"
                    self.log(f"🔧 自动调整: {status}")
                elif key == ord('s'):
                    self.enable_suggestions = not self.enable_suggestions
                    status = "开启" if self.enable_suggestions else "关闭"
                    self.log(f"🔧 智能建议: {status}")
                elif key == ord('c'):
                    self.annotations = []
                    self.log("🗑️  已清除所有标注")
                elif key == ord('u'):
                    if self.annotations:
                        removed = self.annotations.pop()
                        self.log(f"↩️  已撤销: {removed[4]}")
                elif key == ord('h'):
                    self.print_help()
                elif ord('1') <= key <= ord('9'):
                    idx = key - ord('1')
                    if idx < len(self.labels):
                        self.current_label_idx = idx
                        self.log(f"🏷️  切换到: {self.labels[idx]}")
        
        except KeyboardInterrupt:
            self.log("⚠️  用户中断（Ctrl+C）")
        except Exception as e:
            self.log(f"❌ 错误: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.log_fp.close()
            self.log("👋 已退出")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='智能标注工具')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--output', type=str, default='data/smart_annotations',
                       help='输出目录')
    args = parser.parse_args()
    
    annotator = SmartAnnotator(args.output)
    annotator.run(args.camera)

if __name__ == "__main__":
    main()
