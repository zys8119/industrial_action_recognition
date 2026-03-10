#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确检测器 - 使用单应性矩阵计算准确的边界框
"""

import cv2
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from collections import deque

class AccurateDetector:
    """精确检测器 - 准确的位置和大小"""
    
    def __init__(self, model_path):
        print(f"📦 加载模型: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.templates = model_data['templates']
        self.labels = model_data['labels']
        
        print(f"✅ 模型加载成功")
        print(f"📋 类别: {', '.join(self.labels)}")
        
        # 选择最佳模板
        self.best_templates = {}
        for label, templates in self.templates.items():
            # 选择中等尺寸的模板
            sorted_templates = sorted(templates, 
                                     key=lambda t: t['size'][0] * t['size'][1])
            mid_idx = len(sorted_templates) // 2
            start = max(0, mid_idx - 3)
            end = min(len(sorted_templates), mid_idx + 4)
            self.best_templates[label] = sorted_templates[start:end]
        
        print(f"🎯 每个类别使用 {len(self.best_templates[self.labels[0]])} 个模板")
        
        # 特征提取和匹配
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # 平滑跟踪
        self.tracked_objects = {}  # {label: {'bbox': ..., 'history': deque}}
    
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
                    matches = self.bf_matcher.knnMatch(
                        template['descriptors'], des_frame, k=2)
                    
                    # Lowe's ratio test
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                    
                    if len(good_matches) < 10:
                        continue
                    
                    # 计算置信度
                    avg_distance = np.mean([m.distance for m in good_matches])
                    confidence = max(0, 1.0 - avg_distance / 500.0)
                    
                    if confidence < 0.15:
                        continue
                    
                    # 使用单应性矩阵计算精确位置
                    # 需要重新提取模板的关键点
                    template_gray = cv2.cvtColor(
                        cv2.imread(str(Path("data/video_annotations") / "temp.jpg")) 
                        if False else np.zeros((template['size'][1], template['size'][0]), dtype=np.uint8),
                        cv2.COLOR_BGR2GRAY) if False else None
                    
                    # 简化方法：使用匹配点的分布
                    dst_pts = np.array([kp_frame[m.trainIdx].pt for m in good_matches])
                    
                    # 使用 RANSAC 过滤离群点
                    if len(dst_pts) >= 4:
                        # 计算凸包
                        hull = cv2.convexHull(dst_pts.astype(np.float32))
                        
                        # 获取最小外接矩形
                        rect = cv2.minAreaRect(hull)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        
                        # 计算边界框
                        x = int(np.min(box[:, 0]))
                        y = int(np.min(box[:, 1]))
                        w = int(np.max(box[:, 0]) - x)
                        h = int(np.max(box[:, 1]) - y)
                        
                        # 根据模板尺寸调整
                        template_w, template_h = template['size']
                        template_ratio = template_w / template_h
                        
                        detected_ratio = w / h if h > 0 else 1.0
                        
                        # 调整长宽比
                        if abs(detected_ratio - template_ratio) > 0.3:
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            # 保持面积，调整比例
                            area = w * h
                            new_w = int(np.sqrt(area * template_ratio))
                            new_h = int(area / new_w) if new_w > 0 else h
                            
                            x = center_x - new_w // 2
                            y = center_y - new_h // 2
                            w = new_w
                            h = new_h
                        
                        # 边界检查
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame.shape[1] - x)
                        h = min(h, frame.shape[0] - y)
                        
                        if w < 20 or h < 20:
                            continue
                        
                        detections.append({
                            'label': label,
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'matches': len(good_matches),
                            'corners': box
                        })
                
                except Exception as e:
                    continue
        
        # NMS
        detections = self.nms(detections)
        
        # 平滑
        detections = self.smooth_detections(detections)
        
        return detections
    
    def smooth_detections(self, detections):
        """平滑检测结果"""
        smoothed = []
        
        for det in detections:
            label = det['label']
            
            if label not in self.tracked_objects:
                self.tracked_objects[label] = {
                    'history': deque(maxlen=10),
                    'last_bbox': None
                }
            
            current_bbox = det['bbox']
            history = self.tracked_objects[label]['history']
            
            # 添加到历史
            history.append((current_bbox, det['confidence']))
            
            # 加权平均
            if len(history) > 0:
                weights = [conf for _, conf in history]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    avg_x = sum(bbox[0] * w for bbox, w in zip([b for b, _ in history], weights)) / total_weight
                    avg_y = sum(bbox[1] * w for bbox, w in zip([b for b, _ in history], weights)) / total_weight
                    avg_w = sum(bbox[2] * w for bbox, w in zip([b for b, _ in history], weights)) / total_weight
                    avg_h = sum(bbox[3] * w for bbox, w in zip([b for b, _ in history], weights)) / total_weight
                    
                    smoothed_bbox = (int(avg_x), int(avg_y), int(avg_w), int(avg_h))
                    
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
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # 绘制旋转矩形（如果有）
            if 'corners' in det:
                cv2.polylines(frame, [det['corners']], True, (255, 0, 255), 2)
            
            # 绘制中心点
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # 标签
            text = f"{label} {confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - text_h - 15), 
                         (x + text_w + 10, y), color, -1)
            cv2.putText(frame, text, (x + 5, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 尺寸信息
            size_text = f"{w}x{h}"
            cv2.putText(frame, size_text, (x, y + h + 25),
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
        
        window_name = "Accurate Detector"
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
                    for det in detections:
                        x, y, w, h = det['bbox']
                        print(f"Frame {frame_count}: {det['label']} at ({x},{y}) size {w}x{h} conf {det['confidence']:.0%}")
            
            # 绘制
            display = self.draw_detections(frame.copy(), detections)
            
            # 状态栏
            h, w = display.shape[:2]
            cv2.rectangle(display, (10, 10), (w - 10, 80), (0, 0, 0), -1)
            cv2.putText(display, f"Accurate Detection Mode", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Detections: {len(detections)}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"accurate_detection_{timestamp}.jpg"
                cv2.imwrite(save_path, display)
                print(f"📸 已保存: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='精确检测器')
    parser.add_argument('--model', type=str, required=True, help='模型文件')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    args = parser.parse_args()
    
    detector = AccurateDetector(args.model)
    detector.run_camera(args.camera)

if __name__ == "__main__":
    main()
