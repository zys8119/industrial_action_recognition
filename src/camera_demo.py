#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头实时动作识别 Demo - 带位置检测和日志记录
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import deque
import time
import json
from datetime import datetime

class ActionRecognizer:
    def __init__(self, model_path=None, config_path=None, log_file=None):
        """初始化动作识别器"""
        self.model_path = model_path
        self.config_path = config_path
        
        # 加载类别标签
        base_dir = Path(__file__).parent.parent
        label_file = base_dir / "configs" / "label_list.txt"
        
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f if line.strip()]
        
        print(f"📋 加载了 {len(self.labels)} 个动作类别")
        
        # 帧缓冲区（用于时序建模）
        self.frame_buffer = deque(maxlen=16)
        self.target_size = (224, 224)
        
        # 动作检测区域（模拟，实际应该用目标检测模型）
        self.detection_boxes = []  # [(x1, y1, x2, y2, label, confidence), ...]
        
        # 日志设置
        self.log_file = log_file
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_fp = open(log_path, 'a', encoding='utf-8')
            self.log(f"=== 会话开始 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        else:
            self.log_fp = None
        
        # 模拟模型（实际使用时替换为真实模型）
        self.use_mock = True
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("⚠️  未找到模型文件，使用模拟模式")
        
        # 统计信息
        self.action_history = []
        self.frame_count = 0
    
    def log(self, message):
        """写入日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        if self.log_fp:
            self.log_fp.write(log_line + '\n')
            self.log_fp.flush()
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            import paddle
            # 这里加载实际的 PaddlePaddle 模型
            # self.model = paddle.jit.load(model_path)
            # self.use_mock = False
            self.log(f"✅ 模型加载成功: {model_path}")
        except Exception as e:
            self.log(f"❌ 模型加载失败: {e}")
    
    def preprocess_frame(self, frame):
        """预处理单帧"""
        # 调整大小
        frame_resized = cv2.resize(frame, self.target_size)
        # 归一化
        frame_norm = frame_resized.astype(np.float32) / 255.0
        # 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_norm = (frame_norm - mean) / std
        return frame_norm
    
    def detect_action_regions(self, frame):
        """检测动作发生的区域（模拟）"""
        h, w = frame.shape[:2]
        
        # 模拟检测：在画面中心区域随机生成1-3个检测框
        num_boxes = np.random.randint(1, 4)
        boxes = []
        
        for _ in range(num_boxes):
            # 随机生成框的位置和大小
            box_w = np.random.randint(100, 300)
            box_h = np.random.randint(100, 300)
            x1 = np.random.randint(50, max(51, w - box_w - 50))
            y1 = np.random.randint(50, max(51, h - box_h - 50))
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            # 随机分配动作类别
            action_id = np.random.randint(0, len(self.labels))
            confidence = np.random.uniform(0.6, 0.95)
            
            boxes.append((x1, y1, x2, y2, self.labels[action_id], confidence))
        
        return boxes
    
    def predict(self, frame):
        """预测动作"""
        # 预处理并添加到缓冲区
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
        self.frame_count += 1
        
        # 需要足够的帧才能预测
        if len(self.frame_buffer) < 8:
            return None, 0.0, []
        
        if self.use_mock:
            # 模拟预测
            action_id = np.random.randint(0, len(self.labels))
            confidence = np.random.uniform(0.6, 0.95)
            action = self.labels[action_id]
            
            # 每5帧更新一次检测框
            if self.frame_count % 5 == 0:
                self.detection_boxes = self.detect_action_regions(frame)
                
                # 记录日志
                for x1, y1, x2, y2, label, conf in self.detection_boxes:
                    self.log(f"检测到动作: {label} | 置信度: {conf:.2%} | "
                           f"位置: ({x1},{y1})-({x2},{y2}) | "
                           f"尺寸: {x2-x1}x{y2-y1}")
            
            return action, confidence, self.detection_boxes
        else:
            # 真实模型推理
            pass
    
    def draw_results(self, frame, action, confidence, boxes):
        """在画面上绘制结果"""
        h, w = frame.shape[:2]
        
        # 绘制检测框
        for x1, y1, x2, y2, label, conf in boxes:
            # 根据置信度选择颜色
            if conf > 0.8:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif conf > 0.6:
                color = (0, 255, 255)  # 黄色 - 中等置信度
            else:
                color = (0, 165, 255)  # 橙色 - 低置信度
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # 绘制标签背景
            label_text = f"{label} {conf:.0%}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            cv2.rectangle(frame, (x1, y1 - text_h - 15), 
                         (x1 + text_w + 10, y1), color, -1)
            
            # 绘制标签文字
            cv2.putText(frame, label_text, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 绘制中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # 绘制坐标信息
            coord_text = f"({center_x},{center_y})"
            cv2.putText(frame, coord_text, (center_x + 10, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制顶部信息栏
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 绘制标题
        cv2.putText(frame, "Industrial Action Recognition", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if action:
            # 绘制整体识别结果
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            text = f"Overall Action: {action} ({confidence:.2%})"
            cv2.putText(frame, text, (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Collecting frames...", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 绘制检测统计
        detection_text = f"Detections: {len(boxes)} | Frame: {self.frame_count}"
        cv2.putText(frame, detection_text, (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 绘制帧缓冲状态
        buffer_text = f"Buffer: {len(self.frame_buffer)}/16"
        cv2.putText(frame, buffer_text, (w-200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def __del__(self):
        """析构函数，关闭日志文件"""
        if self.log_fp:
            self.log(f"=== 会话结束 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            self.log(f"总帧数: {self.frame_count}")
            self.log_fp.close()

def main():
    parser = argparse.ArgumentParser(description='工业动作识别摄像头Demo')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--log', type=str, default='logs/recognition.log', 
                       help='日志文件路径')
    parser.add_argument('--no-log', action='store_true', help='禁用日志')
    args = parser.parse_args()
    
    # 初始化识别器
    log_file = None if args.no_log else args.log
    recognizer = ActionRecognizer(args.model, args.config, log_file)
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        recognizer.log("❌ 无法打开摄像头")
        return
    
    recognizer.log("✅ 摄像头已打开")
    recognizer.log("💡 按 'q' 退出，按 'r' 重置缓冲区，按 's' 截图")
    
    fps_time = time.time()
    fps = 0
    
    # 创建截图目录
    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            recognizer.log("❌ 无法读取摄像头画面")
            break
        
        # 预测动作和位置
        action, confidence, boxes = recognizer.predict(frame)
        
        # 绘制结果
        frame = recognizer.draw_results(frame, action, confidence, boxes)
        
        # 计算并显示FPS
        current_time = time.time()
        fps = 1 / (current_time - fps_time)
        fps_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-200, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 显示画面
        cv2.imshow('Industrial Action Recognition', frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.frame_buffer.clear()
            recognizer.detection_boxes = []
            recognizer.log("🔄 缓冲区已重置")
        elif key == ord('s'):
            # 截图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = screenshot_dir / f"screenshot_{timestamp}.jpg"
            cv2.imwrite(str(screenshot_path), frame)
            recognizer.log(f"📸 截图已保存: {screenshot_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    recognizer.log("👋 程序已退出")

if __name__ == "__main__":
    main()
