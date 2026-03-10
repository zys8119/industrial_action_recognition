#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于标注数据的物体识别器
读取手动标注的数据，学习物体特征，然后自动识别
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class AnnotationBasedDetector:
    """基于标注数据的检测器"""
    
    def __init__(self, annotations_dir="data/smart_annotations"):
        self.annotations_dir = Path(annotations_dir)
        self.templates = {}  # {label: [template_data, ...]}
        
        # 特征提取器
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # 匹配器
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # 平滑跟踪
        self.tracked_detections = {}  # {label: deque of recent detections}
        self.smooth_window = 5  # 平滑窗口大小
        
        print("🎯 基于标注的检测器已启动")
        self.load_annotations()
    
    def load_annotations(self):
        """加载所有标注数据并提取特征"""
        json_files = list(self.annotations_dir.glob("annotated_*.json"))
        
        if not json_files:
            print(f"⚠️  未找到标注文件，请先运行标注工具")
            print(f"   目录: {self.annotations_dir}")
            return
        
        print(f"📂 找到 {len(json_files)} 个标注文件")
        
        total_templates = 0
        
        for json_file in json_files:
            # 读取JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 读取对应的图像
            img_file = json_file.parent / json_file.name.replace('.json', '.jpg')
            if not img_file.exists():
                continue
            
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # 处理每个标注
            for ann in data['annotations']:
                x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
                label = ann['label']
                
                # 提取ROI
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                # 提取特征
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = self.sift.detectAndCompute(gray_roi, None)
                
                if descriptors is None or len(keypoints) < 5:
                    # 尝试ORB
                    keypoints, descriptors = self.orb.detectAndCompute(gray_roi, None)
                
                if descriptors is None:
                    continue
                
                # 保存模板
                if label not in self.templates:
                    self.templates[label] = []
                
                self.templates[label].append({
                    'image': roi,
                    'gray': gray_roi,
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'size': (w, h),
                    'source': json_file.name
                })
                
                total_templates += 1
        
        print(f"✅ 加载完成:")
        for label, templates in self.templates.items():
            print(f"   {label}: {len(templates)} 个模板")
        print(f"   总计: {total_templates} 个模板")
        
        if total_templates == 0:
            print(f"\n❌ 没有有效的模板！")
            print(f"💡 请确保:")
            print(f"   1. 运行标注工具并保存标注")
            print(f"   2. 标注的物体有清晰的特征")
            print(f"   3. 标注框不要太小")
    
    def calculate_geometry(self, corners):
        """计算物体的几何信息：旋转角度、尺寸等"""
        # corners 是 4 个角点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        pts = corners.reshape(4, 2)
        
        # 计算最小外接矩形（可以旋转）
        rect = cv2.minAreaRect(pts.astype(np.float32))
        center, (width, height), angle = rect
        
        # 确保宽度大于高度
        if width < height:
            width, height = height, width
            angle = angle + 90
        
        # 归一化角度到 [-90, 90]
        if angle > 90:
            angle = angle - 180
        elif angle < -90:
            angle = angle + 180
        
        # 计算面积
        area = width * height
        
        # 计算长宽比
        aspect_ratio = width / height if height > 0 else 0
        
        # 计算中心点
        center_x, center_y = int(center[0]), int(center[1])
        
        return {
            'center': (center_x, center_y),
            'width': float(width),
            'height': float(height),
            'angle': float(angle),
            'area': float(area),
            'aspect_ratio': float(aspect_ratio),
            'rotated_rect': rect
        }
    
    def smooth_detections(self, detections):
        """平滑检测结果，减少抖动"""
        from collections import deque
        
        smoothed = []
        
        for det in detections:
            label = det['label']
            
            # 初始化跟踪队列
            if label not in self.tracked_detections:
                self.tracked_detections[label] = deque(maxlen=self.smooth_window)
            
            # 添加当前检测
            self.tracked_detections[label].append(det['bbox'])
            
            # 计算平均位置
            if len(self.tracked_detections[label]) > 0:
                bboxes = list(self.tracked_detections[label])
                
                # 计算平均坐标
                avg_x = int(np.mean([b[0] for b in bboxes]))
                avg_y = int(np.mean([b[1] for b in bboxes]))
                avg_w = int(np.mean([b[2] for b in bboxes]))
                avg_h = int(np.mean([b[3] for b in bboxes]))
                
                # 创建平滑后的检测
                smoothed_det = det.copy()
                smoothed_det['bbox'] = (avg_x, avg_y, avg_w, avg_h)
                smoothed.append(smoothed_det)
            else:
                smoothed.append(det)
        
        return smoothed
    
    def detect(self, frame, min_matches=8, confidence_threshold=0.5):
        """在帧中检测物体"""
        if not self.templates:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
        
        if des_frame is None:
            return []
        
        detections = []
        
        # 对每个类别的每个模板进行匹配
        for label, templates in self.templates.items():
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
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)
                    
                    if len(good_matches) < min_matches:
                        continue
                    
                    # 提取匹配点
                    src_pts = np.float32([
                        template['keypoints'][m.queryIdx].pt 
                        for m in good_matches
                    ]).reshape(-1, 1, 2)
                    
                    dst_pts = np.float32([
                        kp_frame[m.trainIdx].pt 
                        for m in good_matches
                    ]).reshape(-1, 1, 2)
                    
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
                    
                    # 计算矩形边界
                    x, y, w, h = cv2.boundingRect(dst)
                    
                    # 计算旋转角度和几何信息
                    geometry = self.calculate_geometry(dst)
                    
                    # 计算置信度
                    inliers = np.sum(mask)
                    confidence = inliers / len(good_matches)
                    
                    if confidence < confidence_threshold:
                        continue
                    
                    detections.append({
                        'label': label,
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'matches': len(good_matches),
                        'inliers': int(inliers),
                        'corners': dst,
                        'geometry': geometry  # 新增几何信息
                    })
                
                except Exception as e:
                    continue
        
        # 非极大值抑制（去除重复检测）
        detections = self.nms(detections, iou_threshold=0.5)
        
        # 平滑检测结果
        detections = self.smooth_detections(detections)
        
        return detections
    
    def nms(self, detections, iou_threshold=0.5):
        """非极大值抑制"""
        if len(detections) == 0:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # 移除与best重叠度高的检测
            detections = [
                det for det in detections
                if self.iou(best['bbox'], det['bbox']) < iou_threshold
            ]
        
        return keep
    
    def iou(self, box1, box2):
        """计算IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 计算并集
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def draw_detections(self, frame, detections, show_geometry=True):
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
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # 绘制精确轮廓
            corners = det['corners'].astype(np.int32)
            cv2.polylines(frame, [corners], True, color, 2)
            
            # 绘制旋转矩形
            if show_geometry and 'rotated_rect' in geometry:
                box = cv2.boxPoints(geometry['rotated_rect'])
                box = np.int32(box)
                cv2.drawContours(frame, [box], 0, (255, 0, 255), 2)
            
            # 绘制中心点和坐标轴
            if show_geometry and 'center' in geometry:
                center = geometry['center']
                angle = geometry['angle']
                
                # 绘制中心点
                cv2.circle(frame, center, 5, color, -1)
                cv2.circle(frame, center, 8, color, 2)
                
                # 绘制方向箭头（指示旋转角度）
                length = 50
                angle_rad = np.radians(angle)
                end_x = int(center[0] + length * np.cos(angle_rad))
                end_y = int(center[1] + length * np.sin(angle_rad))
                cv2.arrowedLine(frame, center, (end_x, end_y), 
                              (255, 0, 255), 2, tipLength=0.3)
            
            # 绘制标签
            text = f"{label} {confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            cv2.rectangle(frame, (x, y - text_h - 10), 
                         (x + text_w + 10, y), color, -1)
            cv2.putText(frame, text, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 绘制几何信息
            if show_geometry and geometry:
                info_y = y + h + 20
                
                # 角度
                angle_text = f"Angle: {geometry['angle']:.1f}°"
                cv2.putText(frame, angle_text, (x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 尺寸
                size_text = f"Size: {geometry['width']:.0f}x{geometry['height']:.0f}"
                cv2.putText(frame, size_text, (x, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 面积
                area_text = f"Area: {geometry['area']:.0f}px²"
                cv2.putText(frame, area_text, (x, info_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 长宽比
                ratio_text = f"Ratio: {geometry['aspect_ratio']:.2f}"
                cv2.putText(frame, ratio_text, (x, info_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run_camera(self, camera_id=0):
        """运行摄像头检测"""
        if not self.templates:
            print("❌ 没有可用的模板，无法进行检测")
            return
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
        
        print("✅ 摄像头已打开")
        print("💡 按 'q' 退出，按 's' 截图，按 'r' 重置平滑，按 'g' 切换几何信息显示")
        
        window_name = "Annotation-Based Detector"
        cv2.namedWindow(window_name)
        
        frame_count = 0
        detections = []
        show_geometry = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 每3帧检测一次（提高性能）
            if frame_count % 3 == 0:
                detections = self.detect(frame)
            
            # 绘制结果
            display = self.draw_detections(frame.copy(), detections, show_geometry)
            
            # 绘制状态
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
            
            cv2.putText(display, f"Templates: {sum(len(t) for t in self.templates.values())}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Detections: {len(detections)}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Smoothing: {self.smooth_window} frames", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            geometry_status = "ON" if show_geometry else "OFF"
            cv2.putText(display, f"Geometry: {geometry_status}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"detection_result_{timestamp}.jpg"
                cv2.imwrite(save_path, display)
                
                # 保存检测数据到JSON
                json_path = f"detection_result_{timestamp}.json"
                detection_data = {
                    'timestamp': timestamp,
                    'detections': [
                        {
                            'label': det['label'],
                            'confidence': float(det['confidence']),
                            'bbox': [int(x) for x in det['bbox']],
                            'geometry': {
                                'center': [int(x) for x in det['geometry']['center']],
                                'width': float(det['geometry']['width']),
                                'height': float(det['geometry']['height']),
                                'angle': float(det['geometry']['angle']),
                                'area': float(det['geometry']['area']),
                                'aspect_ratio': float(det['geometry']['aspect_ratio'])
                            }
                        }
                        for det in detections
                    ]
                }
                
                import json
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(detection_data, f, indent=2, ensure_ascii=False)
                
                print(f"📸 已保存: {save_path}")
                print(f"📊 数据已保存: {json_path}")
            elif key == ord('r'):
                self.tracked_detections = {}
                print("🔄 已重置平滑缓存")
            elif key == ord('g'):
                show_geometry = not show_geometry
                status = "开启" if show_geometry else "关闭"
                print(f"📐 几何信息显示: {status}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 检测已停止")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='基于标注数据的物体检测')
    parser.add_argument('--annotations', type=str, 
                       default='data/smart_annotations',
                       help='标注数据目录')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--min-matches', type=int, default=8,
                       help='最少匹配点数')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--smooth', type=int, default=5,
                       help='平滑窗口大小（帧数）')
    args = parser.parse_args()
    
    detector = AnnotationBasedDetector(args.annotations)
    detector.smooth_window = args.smooth
    detector.run_camera(args.camera)

if __name__ == "__main__":
    main()
