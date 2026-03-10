#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时图像标注与动态跟踪工具
支持手动标记后自动跟踪物体运动
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import deque

class TrackedObject:
    """被跟踪的物体"""
    def __init__(self, bbox, label, tracker_type='CSRT'):
        self.bbox = bbox  # (x, y, w, h)
        self.label = label
        self.confidence = 1.0
        self.trajectory = deque(maxlen=30)  # 轨迹点
        self.lost_frames = 0
        self.max_lost_frames = 10
        self.is_active = True
        
        # 创建跟踪器
        self.tracker = self.create_tracker(tracker_type)
        
    def create_tracker(self, tracker_type):
        """创建OpenCV跟踪器"""
        try:
            # OpenCV 4.5.1+ 新API
            if tracker_type == 'CSRT':
                return cv2.legacy.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                return cv2.legacy.TrackerKCF_create()
            elif tracker_type == 'MOSSE':
                return cv2.legacy.TrackerMOSSE_create()
            else:
                return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            # OpenCV 旧API
            if tracker_type == 'CSRT':
                return cv2.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                return cv2.TrackerKCF_create()
            elif tracker_type == 'MOSSE':
                return cv2.TrackerMOSSE_create()
            else:
                return cv2.TrackerCSRT_create()
    
    def update(self, frame):
        """更新跟踪"""
        success, bbox = self.tracker.update(frame)
        
        if success:
            self.bbox = tuple(map(int, bbox))
            x, y, w, h = self.bbox
            center = (x + w // 2, y + h // 2)
            self.trajectory.append(center)
            self.lost_frames = 0
            self.confidence = max(0.5, self.confidence - 0.01)
        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.is_active = False
        
        return success

class ImageAnnotatorWithTracking:
    def __init__(self, output_dir="data/annotations", tracker_type='CSRT'):
        """初始化标注与跟踪工具"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标注状态
        self.drawing = False
        self.start_point = None
        self.current_box = None
        
        # 跟踪对象列表
        self.tracked_objects = []
        
        # 当前帧
        self.current_frame = None
        self.frame_count = 0
        
        # 类别标签
        self.labels = self.load_labels()
        self.current_label_idx = 0
        
        # 跟踪器类型
        self.tracker_type = tracker_type
        
        # 显示选项
        self.show_trajectory = True
        self.show_labels = True
        self.show_confidence = True
        
        # 日志
        self.log_file = self.output_dir / "tracking_log.txt"
        self.log_fp = open(self.log_file, 'a', encoding='utf-8')
        
        print("🎨 图像标注与跟踪工具已启动")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"🎯 跟踪器: {tracker_type}")
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
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        self.log_fp.write(log_line + '\n')
        self.log_fp.flush()
    
    def print_help(self):
        """打印帮助信息"""
        print("\n⌨️  快捷键:")
        print("   鼠标拖动    - 标记物体（自动开始跟踪）")
        print("   数字键 1-9  - 切换类别标签")
        print("   SPACE       - 暂停/继续跟踪")
        print("   c           - 清除所有跟踪")
        print("   d           - 删除最后一个跟踪")
        print("   t           - 切换轨迹显示")
        print("   l           - 切换标签显示")
        print("   s           - 保存当前帧")
        print("   r           - 切换跟踪器类型")
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
            if self.drawing and self.start_point and self.current_frame is not None:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # 确保坐标正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                w, h = x2 - x1, y2 - y1
                
                # 只保存有效的框（面积 > 100）
                if w * h > 100:
                    label = self.labels[self.current_label_idx]
                    
                    # 创建跟踪对象
                    tracked_obj = TrackedObject((x1, y1, w, h), label, self.tracker_type)
                    tracked_obj.tracker.init(self.current_frame, (x1, y1, w, h))
                    self.tracked_objects.append(tracked_obj)
                    
                    self.log(f"✅ 添加跟踪: {label} at ({x1},{y1}) size {w}x{h}")
                    print(f"🎯 开始跟踪 #{len(self.tracked_objects)}: {label}")
                
                self.current_box = None
                self.start_point = None
    
    def update_tracking(self, frame):
        """更新所有跟踪对象"""
        active_objects = []
        
        for i, obj in enumerate(self.tracked_objects):
            if obj.is_active:
                success = obj.update(frame)
                if success:
                    active_objects.append(obj)
                else:
                    self.log(f"❌ 跟踪丢失: {obj.label} (ID: {i})")
            else:
                self.log(f"⚠️  跟踪失效: {obj.label} (ID: {i})")
        
        self.tracked_objects = active_objects
    
    def draw_tracked_objects(self, frame):
        """绘制跟踪对象"""
        for i, obj in enumerate(self.tracked_objects):
            x, y, w, h = obj.bbox
            color = self.get_label_color(obj.label)
            
            # 根据置信度调整颜色亮度
            color = tuple(int(c * obj.confidence) for c in color)
            
            # 绘制边界框
            thickness = 3 if obj.confidence > 0.7 else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # 绘制标签
            if self.show_labels:
                label_text = f"#{i+1} {obj.label}"
                if self.show_confidence:
                    label_text += f" {obj.confidence:.0%}"
                
                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - text_h - 10), 
                            (x + text_w + 10, y), color, -1)
                cv2.putText(frame, label_text, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 绘制中心点
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
            
            # 绘制轨迹
            if self.show_trajectory and len(obj.trajectory) > 1:
                points = np.array(list(obj.trajectory), dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
                
                # 绘制轨迹点
                for point in obj.trajectory:
                    cv2.circle(frame, point, 2, color, -1)
    
    def draw_current_box(self, frame):
        """绘制正在绘制的框"""
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            label = self.labels[self.current_label_idx]
            color = self.get_label_color(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def get_label_color(self, label):
        """根据标签获取颜色"""
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
            (128, 255, 0),  # 春绿
            (255, 128, 0),  # 橙色
        ]
        idx = self.labels.index(label) if label in self.labels else 0
        return colors[idx % len(colors)]
    
    def draw_status_bar(self, frame):
        """绘制状态栏"""
        h, w = frame.shape[:2]
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 当前类别
        current_label = self.labels[self.current_label_idx]
        color = self.get_label_color(current_label)
        cv2.putText(frame, f"Label: {current_label} (1-{len(self.labels)})",
                   (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 跟踪统计
        active_count = len(self.tracked_objects)
        cv2.putText(frame, f"Tracking: {active_count} objects | Frame: {self.frame_count}",
                   (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示选项
        options = []
        if self.show_trajectory:
            options.append("Trajectory")
        if self.show_labels:
            options.append("Labels")
        options_text = " | ".join(options) if options else "None"
        cv2.putText(frame, f"Display: {options_text}",
                   (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 快捷键提示
        cv2.putText(frame, "SPACE:Pause | C:Clear | D:Delete | T:Trajectory | S:Save | Q:Quit",
                   (w - 700, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def save_frame(self):
        """保存当前帧和跟踪数据"""
        if self.current_frame is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 保存图像
        img_filename = f"tracked_{timestamp}.jpg"
        img_path = self.output_dir / img_filename
        
        # 绘制所有跟踪信息
        save_frame = self.current_frame.copy()
        self.draw_tracked_objects(save_frame)
        cv2.imwrite(str(img_path), save_frame)
        
        # 保存跟踪数据
        tracking_data = {
            "timestamp": timestamp,
            "frame": int(self.frame_count),
            "objects": [
                {
                    "id": int(i),
                    "label": str(obj.label),
                    "bbox": [int(obj.bbox[0]), int(obj.bbox[1]), 
                            int(obj.bbox[2]), int(obj.bbox[3])],
                    "confidence": float(obj.confidence),
                    "trajectory": [[int(p[0]), int(p[1])] for p in obj.trajectory],
                    "trajectory_length": int(len(obj.trajectory))
                }
                for i, obj in enumerate(self.tracked_objects)
            ]
        }
        
        json_filename = f"tracked_{timestamp}.json"
        json_path = self.output_dir / json_filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 已保存: {img_filename} ({len(self.tracked_objects)} 个跟踪对象)")
    
    def run(self, camera_id=0):
        """运行标注与跟踪工具"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
        
        window_name = "Annotator with Tracking"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.log("✅ 摄像头已打开，开始标注与跟踪...")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                
                self.current_frame = frame.copy()
                self.frame_count += 1
                
                # 更新跟踪
                if self.tracked_objects:
                    self.update_tracking(frame)
            else:
                frame = self.current_frame.copy()
            
            # 绘制跟踪对象
            self.draw_tracked_objects(frame)
            
            # 绘制正在绘制的框
            self.draw_current_box(frame)
            
            # 绘制状态栏
            self.draw_status_bar(frame)
            
            # 显示暂停状态
            if paused:
                cv2.putText(frame, "PAUSED", (frame.shape[1] // 2 - 80, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # 显示
            cv2.imshow(window_name, frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' '):  # SPACE - 暂停/继续
                paused = not paused
                status = "暂停" if paused else "继续"
                self.log(f"⏸️  {status}跟踪")
            
            elif key == ord('c'):  # 清除所有
                self.tracked_objects = []
                self.log("🗑️  已清除所有跟踪")
            
            elif key == ord('d'):  # 删除最后一个
                if self.tracked_objects:
                    removed = self.tracked_objects.pop()
                    self.log(f"↩️  已删除: {removed.label}")
            
            elif key == ord('t'):  # 切换轨迹显示
                self.show_trajectory = not self.show_trajectory
                status = "开启" if self.show_trajectory else "关闭"
                self.log(f"🎨 轨迹显示: {status}")
            
            elif key == ord('l'):  # 切换标签显示
                self.show_labels = not self.show_labels
                status = "开启" if self.show_labels else "关闭"
                self.log(f"🏷️  标签显示: {status}")
            
            elif key == ord('s'):  # 保存
                self.save_frame()
            
            elif key == ord('r'):  # 切换跟踪器
                trackers = ['CSRT', 'KCF', 'MOSSE']
                current_idx = trackers.index(self.tracker_type)
                self.tracker_type = trackers[(current_idx + 1) % len(trackers)]
                self.log(f"🔄 切换跟踪器: {self.tracker_type}")
            
            elif key == ord('h'):  # 帮助
                self.print_help()
            
            elif ord('1') <= key <= ord('9'):  # 切换类别
                idx = key - ord('1')
                if idx < len(self.labels):
                    self.current_label_idx = idx
                    self.log(f"🏷️  切换到类别: {self.labels[idx]}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.log_fp.close()
        
        print("👋 标注与跟踪工具已退出")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='实时图像标注与动态跟踪工具')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--output', type=str, default='data/tracking', help='输出目录')
    parser.add_argument('--tracker', type=str, default='CSRT',
                       choices=['CSRT', 'KCF', 'MOSSE'],
                       help='跟踪器类型 (CSRT=精确但慢, KCF=平衡, MOSSE=快速但不精确)')
    args = parser.parse_args()
    
    annotator = ImageAnnotatorWithTracking(args.output, args.tracker)
    annotator.run(args.camera)

if __name__ == "__main__":
    main()
