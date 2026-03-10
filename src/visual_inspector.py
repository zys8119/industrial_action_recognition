#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉检查工具 - 基于参考图片的目标检测和训练
提供一张参考图片，自动学习并检测相似目标
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse

class VisualInspector:
    """视觉检查器 - 基于模板匹配和特征学习"""
    
    def __init__(self, output_dir="data/inspection"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 参考模板
        self.templates = []  # [(name, image, features), ...]
        
        # 特征检测器
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        
        # 匹配器
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # 检测结果
        self.detections = []
        
        # 日志
        self.log_file = self.output_dir / "inspection_log.txt"
        self.log_fp = open(self.log_file, 'a', encoding='utf-8')
        
        print("🔍 视觉检查工具已启动")
        self.log("=== 会话开始 ===")
    
    def log(self, message):
        """写入日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        self.log_fp.write(log_line + '\n')
        self.log_fp.flush()
    
    def add_template(self, image_path, name=None):
        """添加参考模板"""
        image_path = Path(image_path)
        if not image_path.exists():
            self.log(f"❌ 图片不存在: {image_path}")
            return False
        
        # 读取图片
        template = cv2.imread(str(image_path))
        if template is None:
            self.log(f"❌ 无法读取图片: {image_path}")
            return False
        
        # 转换为灰度图
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # 提取特征
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 10:
            self.log(f"⚠️  特征点太少，尝试使用 ORB")
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None:
            self.log(f"❌ 无法提取特征")
            return False
        
        # 保存模板
        template_name = name or image_path.stem
        self.templates.append({
            'name': template_name,
            'image': template,
            'gray': gray,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'size': template.shape[:2]
        })
        
        self.log(f"✅ 添加模板: {template_name} ({len(keypoints)} 个特征点)")
        
        # 保存特征可视化
        vis_image = cv2.drawKeypoints(template, keypoints, None, 
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        vis_path = self.output_dir / f"template_{template_name}_features.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        self.log(f"💾 特征可视化已保存: {vis_path}")
        
        return True
    
    def detect_in_frame(self, frame, min_matches=10, match_threshold=0.7):
        """在帧中检测目标"""
        if not self.templates:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 提取帧的特征
        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
        
        if des_frame is None:
            return []
        
        detections = []
        
        # 对每个模板进行匹配
        for template in self.templates:
            # 特征匹配
            matches = self.bf_matcher.knnMatch(template['descriptors'], des_frame, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < match_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < min_matches:
                continue
            
            # 提取匹配点
            src_pts = np.float32([template['keypoints'][m.queryIdx].pt 
                                 for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt 
                                 for m in good_matches]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                continue
            
            # 计算边界框
            h, w = template['size']
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            # 计算边界框的矩形
            x, y, w, h = cv2.boundingRect(dst)
            
            # 计算置信度（基于内点数量）
            inliers = np.sum(mask)
            confidence = inliers / len(good_matches)
            
            detections.append({
                'name': template['name'],
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'matches': len(good_matches),
                'inliers': inliers,
                'corners': dst
            })
            
            self.log(f"🎯 检测到: {template['name']} | "
                    f"置信度: {confidence:.2%} | "
                    f"匹配点: {len(good_matches)} | "
                    f"位置: ({x},{y}) 尺寸: {w}x{h}")
        
        return detections
    
    def draw_detections(self, frame, detections):
        """绘制检测结果"""
        for det in detections:
            # 绘制边界框
            x, y, w, h = det['bbox']
            
            # 根据置信度选择颜色
            if det['confidence'] > 0.7:
                color = (0, 255, 0)  # 绿色
            elif det['confidence'] > 0.5:
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 165, 255)  # 橙色
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # 绘制多边形（更精确的轮廓）
            corners = det['corners'].astype(np.int32)
            cv2.polylines(frame, [corners], True, color, 2)
            
            # 绘制标签
            label = f"{det['name']} {det['confidence']:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            cv2.rectangle(frame, (x, y - text_h - 10), 
                         (x + text_w + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 绘制匹配点数量
            info_text = f"Matches: {det['matches']}"
            cv2.putText(frame, info_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run_camera(self, camera_id=0):
        """运行摄像头检测"""
        if not self.templates:
            self.log("❌ 请先添加参考模板")
            return
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.log("❌ 无法打开摄像头")
            return
        
        self.log("✅ 摄像头已打开")
        
        window_name = "Visual Inspector"
        cv2.namedWindow(window_name)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 每5帧检测一次（提高性能）
            if frame_count % 5 == 0:
                detections = self.detect_in_frame(frame)
                self.detections = detections
            
            # 绘制检测结果
            display = self.draw_detections(frame.copy(), self.detections)
            
            # 绘制状态信息
            h, w = display.shape[:2]
            cv2.rectangle(display, (10, 10), (w - 10, 80), (0, 0, 0), -1)
            cv2.putText(display, f"Templates: {len(self.templates)}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Detections: {len(self.detections)}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(save_path), display)
                self.log(f"📸 已保存: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.log("👋 检测已停止")
    
    def inspect_image(self, image_path):
        """检查单张图片"""
        image_path = Path(image_path)
        if not image_path.exists():
            self.log(f"❌ 图片不存在: {image_path}")
            return
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            self.log(f"❌ 无法读取图片: {image_path}")
            return
        
        self.log(f"🔍 检查图片: {image_path}")
        
        # 检测
        detections = self.detect_in_frame(frame)
        
        # 绘制结果
        result = self.draw_detections(frame.copy(), detections)
        
        # 保存结果
        output_path = self.output_dir / f"inspected_{image_path.name}"
        cv2.imwrite(str(output_path), result)
        self.log(f"💾 结果已保存: {output_path}")
        
        # 显示结果
        cv2.imshow("Inspection Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return detections
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'log_fp') and self.log_fp:
            self.log("=== 会话结束 ===")
            self.log_fp.close()

def main():
    parser = argparse.ArgumentParser(description='视觉检查工具')
    parser.add_argument('--template', type=str, required=True,
                       help='参考模板图片路径')
    parser.add_argument('--name', type=str, help='模板名称')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--image', type=str, help='检查单张图片')
    parser.add_argument('--output', type=str, default='data/inspection',
                       help='输出目录')
    args = parser.parse_args()
    
    # 创建检查器
    inspector = VisualInspector(args.output)
    
    # 添加模板
    if not inspector.add_template(args.template, args.name):
        return
    
    # 运行检测
    if args.image:
        inspector.inspect_image(args.image)
    else:
        inspector.run_camera(args.camera)

if __name__ == "__main__":
    main()
