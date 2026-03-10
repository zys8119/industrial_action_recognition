#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的模型进行检测
"""

import cv2
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from collections import deque

class VideoModelDetector:
    """基于视频训练模型的检测器"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        # 加载模型
        print(f"📦 加载模型: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.templates = model_data['templates']
        self.labels = model_data['labels']
        
        print(f"✅ 模型加载成功")
        print(f"📋 类别: {', '.join(self.labels)}")
        print(f"📊 样本数: {model_data['total_samples']}")
        
        # 特征提取器和匹配器
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # 平滑跟踪
        self.tracked_detections = {}
        self.smooth_window = 5
    
    def detect(self, frame, min_matches=8, confidence_threshold=0.5):
        """检测物体"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
        
        if des_frame is None:
            return []
        
        detections = []
        
        # 对每个类别的每个模板进行匹配
        for label, templates in self.templates.items():
            for template in templates:
                try:
                    matches = self.bf_matcher.knnMatch(
                        template['descriptors'], des_frame, k=2)
                    
                    # Lowe's ratio test
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)
                    
                    if len(good_matches) < min_matches:
                        continue
                    
                    # 提取匹配点（使用索引）
                    src_pts = []
                    dst_pts = []
                    
                    for m in good_matches:
                        # 从模板的描述符索引获取对应的关键点
                        # 注意：我们需要重新计算关键点位置
                        # 这里使用简化方法：假设特征均匀分布
                        template_h, template_w = template['size'][1], template['size'][0]
                        
                        # 使用描述符索引估算位置（简化）
                        src_x = (m.queryIdx % template_w) * (template_w / 100.0)
                        src_y = (m.queryIdx // template_w) * (template_h / 100.0)
                        src_pts.append([src_x, src_y])
                        
                        # 目标点使用实际关键点
                        dst_pts.append(kp_frame[m.trainIdx].pt)
                    
                    if len(src_pts) < min_matches:
                        continue
                    
                    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
                    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
                    
                    # 计算单应性矩阵
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if M is None:
                        continue
                    
                    # 计算边界框
                    h, w = template['size'][1], template['size'][0]
                    pts = np.float32([
                        [0, 0], [0, h], [w, h], [w, 0]
                    ]).reshape(-1, 1, 2)
                    
                    dst = cv2.perspectiveTransform(pts, M)
                    x, y, w, h = cv2.boundingRect(dst)
                    
                    # 计算置信度
                    inliers = np.sum(mask)
                    confidence = inliers / len(good_matches)
                    
                    if confidence < confidence_threshold:
                        continue
                    
                    # 计算几何信息
                    geometry = self.calculate_geometry(dst)
                    
                    detections.append({
                        'label': label,
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'matches': len(good_matches),
                        'geometry': geometry
                    })
                
                except Exception:
                    continue
        
        # 非极大值抑制
        detections = self.nms(detections)
        
        # 平滑
        detections = self.smooth_detections(detections)
        
        return detections
    
    def calculate_geometry(self, corners):
        """计算几何信息"""
        pts = corners.reshape(4, 2)
        rect = cv2.minAreaRect(pts.astype(np.float32))
        center, (width, height), angle = rect
        
        if width < height:
            width, height = height, width
            angle = angle + 90
        
        if angle > 90:
            angle = angle - 180
        elif angle < -90:
            angle = angle + 180
        
        return {
            'center': (int(center[0]), int(center[1])),
            'width': float(width),
            'height': float(height),
            'angle': float(angle),
            'area': float(width * height),
            'aspect_ratio': float(width / height if height > 0 else 0)
        }
    
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
    
    def smooth_detections(self, detections):
        """平滑检测结果"""
        smoothed = []
        
        for det in detections:
            label = det['label']
            
            if label not in self.tracked_detections:
                self.tracked_detections[label] = deque(maxlen=self.smooth_window)
            
            self.tracked_detections[label].append(det['bbox'])
            
            if len(self.tracked_detections[label]) > 0:
                bboxes = list(self.tracked_detections[label])
                avg_x = int(np.mean([b[0] for b in bboxes]))
                avg_y = int(np.mean([b[1] for b in bboxes]))
                avg_w = int(np.mean([b[2] for b in bboxes]))
                avg_h = int(np.mean([b[3] for b in bboxes]))
                
                smoothed_det = det.copy()
                smoothed_det['bbox'] = (avg_x, avg_y, avg_w, avg_h)
                smoothed.append(smoothed_det)
            else:
                smoothed.append(det)
        
        return smoothed
    
    def draw_detections(self, frame, detections):
        """绘制检测结果"""
        for det in detections:
            x, y, w, h = det['bbox']
            label = det['label']
            confidence = det['confidence']
            geometry = det.get('geometry', {})
            
            # 颜色
            if confidence > 0.7:
                color = (0, 255, 0)
            elif confidence > 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            # 边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # 标签
            text = f"{label} {confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - text_h - 10), 
                         (x + text_w + 10, y), color, -1)
            cv2.putText(frame, text, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 几何信息
            if geometry:
                info_y = y + h + 20
                cv2.putText(frame, f"Angle: {geometry['angle']:.1f}°", 
                           (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, f"Size: {geometry['width']:.0f}x{geometry['height']:.0f}", 
                           (x, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run_camera(self, camera_id=0):
        """运行摄像头检测"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
        
        print("✅ 摄像头已打开")
        print("💡 按 'q' 退出，按 's' 截图")
        
        window_name = "Video Model Detector"
        cv2.namedWindow(window_name)
        
        frame_count = 0
        detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 每3帧检测一次
            if frame_count % 3 == 0:
                detections = self.detect(frame)
            
            # 绘制
            display = self.draw_detections(frame.copy(), detections)
            
            # 状态栏
            h, w = display.shape[:2]
            cv2.rectangle(display, (10, 10), (w - 10, 80), (0, 0, 0), -1)
            cv2.putText(display, f"Model: {self.model_path.name}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Detections: {len(detections)}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='使用训练好的模型进行检测')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    args = parser.parse_args()
    
    detector = VideoModelDetector(args.model)
    detector.run_camera(args.camera)

if __name__ == "__main__":
    main()
