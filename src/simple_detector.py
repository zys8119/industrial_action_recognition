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
        
        # 平滑
        self.detection_history = deque(maxlen=5)
    
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
                    
                    # 计算边界框
                    xs = [p[0] for p in dst_pts]
                    ys = [p[1] for p in dst_pts]
                    
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # 过滤不合理的框
                    if w < 20 or h < 20 or w > frame.shape[1] * 0.8 or h > frame.shape[0] * 0.8:
                        continue
                    
                    # 扩展边界框（增加一些边距）
                    margin = 10
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(frame.shape[1], x_max + margin)
                    y_max = min(frame.shape[0], y_max + margin)
                    
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
        
        # 记录历史
        self.detection_history.append(detections)
        
        return detections
    
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
    
    def draw_detections(self, frame, detections):
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
            
            # 边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # 标签
            text = f"{label} {confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - text_h - 15), 
                         (x + text_w + 10, y), color, -1)
            cv2.putText(frame, text, (x + 5, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 匹配点数
            info = f"Matches: {det['matches']}"
            cv2.putText(frame, info, (x, y + h + 25),
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
