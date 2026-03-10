#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版检测器 - 使用模板匹配
"""

import cv2
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from collections import deque

class SimpleDetector:
    """简化的检测器 - 使用模板匹配"""
    
    def __init__(self, model_path):
        print(f"📦 加载模型: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.templates = model_data['templates']
        self.labels = model_data['labels']
        
        print(f"✅ 模型加载成功")
        print(f"📋 类别: {', '.join(self.labels)}")
        print(f"📊 样本数: {model_data['total_samples']}")
        
        # 为每个类别选择最好的模板（取前5个）
        self.best_templates = {}
        for label, templates in self.templates.items():
            # 选择尺寸适中的模板
            sorted_templates = sorted(templates, 
                                     key=lambda t: t['size'][0] * t['size'][1])
            mid_idx = len(sorted_templates) // 2
            start = max(0, mid_idx - 2)
            end = min(len(sorted_templates), mid_idx + 3)
            self.best_templates[label] = sorted_templates[start:end]
        
        print(f"🎯 每个类别使用 {len(self.best_templates[self.labels[0]])} 个最佳模板")
        
        # SIFT 匹配器
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # 改进的平滑跟踪
        self.detection_history = {}  # {label: deque of (bbox, confidence)}
        self.smooth_window = 8  # 增加平滑窗口
    
    def detect(self, frame):
        """检测物体"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
        
        if des_frame is None or len(kp_frame) < 10:
            return []
        
        detections = []
        
        for label, templates in self.best_templates.items():
            for template in templates:
                try:
                    # 特征匹配
                    matches = self.bf_matcher.match(
                        template['descriptors'], des_frame)
                    
                    # 按距离排序
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # 取最好的匹配
                    good_matches = matches[:min(50, len(matches))]
                    
                    if len(good_matches) < 8:  # 从 10 改为 8
                        continue
                    
                    # 计算平均匹配质量
                    avg_distance = np.mean([m.distance for m in good_matches])
                    
                    # 距离越小越好，转换为置信度
                    # SIFT 特征距离范围很大，使用更宽松的阈值
                    confidence = max(0, 1.0 - avg_distance / 500.0)  # 改为 500
                    
                    print(f"      [{label}] 匹配: {len(good_matches)} 点, 距离: {avg_distance:.1f}, 置信度: {confidence:.2%}")
                    
                    if confidence < 0.1:  # 改为 0.1，非常宽松
                        continue
                    
                    # 计算检测框（使用匹配点的分布）
                    dst_pts = [kp_frame[m.trainIdx].pt for m in good_matches]
                    
                    if len(dst_pts) < 4:
                        continue
                    
                    # 使用凸包来计算更准确的边界
                    dst_pts_array = np.array(dst_pts, dtype=np.float32)
                    hull = cv2.convexHull(dst_pts_array)
                    
                    # 计算凸包的边界框
                    x_min = int(np.min(hull[:, 0, 0]))
                    y_min = int(np.min(hull[:, 0, 1]))
                    x_max = int(np.max(hull[:, 0, 0]))
                    y_max = int(np.max(hull[:, 0, 1]))
                    
                    # 根据模板的长宽比调整边界框
                    template_w, template_h = template['size']
                    template_ratio = template_w / template_h if template_h > 0 else 1.0
                    
                    detected_w = x_max - x_min
                    detected_h = y_max - y_min
                    detected_ratio = detected_w / detected_h if detected_h > 0 else 1.0
                    
                    # 如果长宽比差异太大，调整边界框
                    if abs(detected_ratio - template_ratio) > 0.5:
                        # 保持中心点，按模板比例调整
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2
                        
                        if detected_ratio > template_ratio:
                            # 检测框太宽，调整宽度
                            new_w = int(detected_h * template_ratio)
                            x_min = center_x - new_w // 2
                            x_max = center_x + new_w // 2
                        else:
                            # 检测框太高，调整高度
                            new_h = int(detected_w / template_ratio)
                            y_min = center_y - new_h // 2
                            y_max = center_y + new_h // 2
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # 过滤不合理的框
                    if w < 20 or h < 20 or w > frame.shape[1] * 0.8 or h > frame.shape[0] * 0.8:
                        continue
                    
                    # 添加适当的边距（根据模板大小）
                    margin_ratio = 0.1  # 10% 边距
                    margin_w = int(w * margin_ratio)
                    margin_h = int(h * margin_ratio)
                    
                    x_min = max(0, x_min - margin_w)
                    y_min = max(0, y_min - margin_h)
                    x_max = min(frame.shape[1], x_max + margin_w)
                    y_max = min(frame.shape[0], y_max + margin_h)
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    detections.append({
                        'label': label,
                        'bbox': (x_min, y_min, w, h),
                        'confidence': confidence,
                        'matches': len(good_matches)
                    })
                
                except Exception as e:
                    continue
        
        # NMS
        detections = self.nms(detections)
        
        # 改进的平滑
        detections = self.smooth_detections_v2(detections)
        
        return detections
    
    def smooth_detections_v2(self, detections):
        """改进的平滑算法"""
        smoothed = []
        
        for det in detections:
            label = det['label']
            bbox = det['bbox']
            confidence = det['confidence']
            
            # 初始化历史
            if label not in self.detection_history:
                self.detection_history[label] = deque(maxlen=self.smooth_window)
            
            # 添加当前检测
            self.detection_history[label].append((bbox, confidence))
            
            # 加权平均（置信度高的权重大）
            history = list(self.detection_history[label])
            
            if len(history) > 0:
                total_weight = 0
                weighted_x = 0
                weighted_y = 0
                weighted_w = 0
                weighted_h = 0
                
                for hist_bbox, hist_conf in history:
                    weight = hist_conf  # 使用置信度作为权重
                    total_weight += weight
                    weighted_x += hist_bbox[0] * weight
                    weighted_y += hist_bbox[1] * weight
                    weighted_w += hist_bbox[2] * weight
                    weighted_h += hist_bbox[3] * weight
                
                if total_weight > 0:
                    smoothed_bbox = (
                        int(weighted_x / total_weight),
                        int(weighted_y / total_weight),
                        int(weighted_w / total_weight),
                        int(weighted_h / total_weight)
                    )
                    
                    smoothed_det = det.copy()
                    smoothed_det['bbox'] = smoothed_bbox
                    smoothed.append(smoothed_det)
                else:
                    smoothed.append(det)
            else:
                smoothed.append(det)
        
        return smoothed
    
    def nms(self, detections, iou_threshold=0.5):
        """非极大值抑制"""
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                det for det in detections
                if self.iou(best['bbox'], det['bbox']) < iou_threshold
            ]
        
        return keep
    
    def iou(self, box1, box2):
        """计算IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def draw_detections(self, frame, detections, show_details=True):
        """绘制检测结果"""
        for det in detections:
            x, y, w, h = det['bbox']
            label = det['label']
            confidence = det['confidence']
            
            # 颜色
            if confidence > 0.6:
                color = (0, 255, 0)
            elif confidence > 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            # 边界框（加粗）
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
            
            # 中心十字线
            center_x = x + w // 2
            center_y = y + h // 2
            cross_size = 10
            cv2.line(frame, (center_x - cross_size, center_y), 
                    (center_x + cross_size, center_y), color, 2)
            cv2.line(frame, (center_x, center_y - cross_size), 
                    (center_x, center_y + cross_size), color, 2)
            
            # 标签（更大更清晰）
            text = f"{label} {confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x, y - text_h - 15), 
                         (x + text_w + 10, y), color, -1)
            cv2.putText(frame, text, (x + 5, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            if show_details:
                # 尺寸信息
                size_text = f"{w}x{h}px"
                cv2.putText(frame, size_text, (x, y + h + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 匹配点数
                matches_text = f"Matches: {det['matches']}"
                cv2.putText(frame, matches_text, (x, y + h + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def run_camera(self, camera_id=0):
        """运行摄像头检测"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
        
        print("✅ 摄像头已打开")
        print("💡 按 'q' 退出，按 's' 截图")
        print("🔍 正在检测...")
        
        window_name = "Simple Detector"
        cv2.namedWindow(window_name)
        
        frame_count = 0
        detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 每2帧检测一次
            if frame_count % 2 == 0:
                detections = self.detect(frame)
                if detections:
                    print(f"Frame {frame_count}: 检测到 {len(detections)} 个物体")
            
            # 绘制
            display = self.draw_detections(frame.copy(), detections)
            
            # 状态栏
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, 100), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
            
            cv2.putText(display, f"Classes: {', '.join(self.labels)}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Detections: {len(detections)}", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 提示
            if len(detections) == 0:
                cv2.putText(display, "No detections - Try moving object closer", 
                           (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"detection_{timestamp}.jpg"
                cv2.imwrite(save_path, display)
                print(f"📸 已保存: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 检测已停止")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='简化版检测器')
    parser.add_argument('--model', type=str, required=True, help='模型文件')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    args = parser.parse_args()
    
    detector = SimpleDetector(args.model)
    detector.run_camera(args.camera)

if __name__ == "__main__":
    main()
