#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频采集自动标注工具
第一次手动框选后，自动跟踪并连续标注物体
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from collections import deque

class VideoAutoAnnotator:
    """视频自动标注器 - 手动框选后自动跟踪标注"""
    
    def __init__(self, output_dir="data/video_annotations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标注状态
        self.drawing = False
        self.start_point = None
        self.current_box = None
        
        # 跟踪对象
        self.tracked_objects = []  # [(tracker, label, id, history), ...]
        self.next_id = 1
        
        # 类别
        self.labels = self.load_labels()
        self.current_label_idx = 0
        
        # 录制状态
        self.recording = False
        self.video_writer = None
        self.frame_annotations = []  # 每帧的标注数据
        
        # 当前帧
        self.current_frame = None
        self.frame_count = 0
        
        # 显示选项
        self.show_trajectory = True
        self.show_id = True
        
        # 日志
        self.log_file = self.output_dir / "auto_annotation_log.txt"
        self.log_fp = open(self.log_file, 'a', encoding='utf-8')
        
        print("🎬 视频自动标注工具已启动")
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
        print("   鼠标拖动    - 框选物体（自动开始跟踪）")
        print("   数字键 1-9  - 切换类别")
        print("   SPACE       - 开始/停止录制")
        print("   T           - 切换轨迹显示")
        print("   I           - 切换ID显示")
        print("   C           - 清除所有跟踪")
        print("   D           - 删除最后一个跟踪")
        print("   S           - 保存当前标注")
        print("   H           - 帮助")
        print("   Q           - 退出")
        print()
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
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
                
                if w * h > 100:
                    label = self.labels[self.current_label_idx]
                    self.add_tracker(x1, y1, w, h, label)
                
                self.current_box = None
                self.start_point = None
    
    def add_tracker(self, x, y, w, h, label):
        """添加跟踪器"""
        # 创建跟踪器（兼容不同版本的 OpenCV）
        tracker = None
        
        # 尝试不同的跟踪器
        tracker_methods = [
            ('CSRT (legacy)', lambda: cv2.legacy.TrackerCSRT_create()),
            ('CSRT', lambda: cv2.TrackerCSRT_create()),
            ('KCF (legacy)', lambda: cv2.legacy.TrackerKCF_create()),
            ('KCF', lambda: cv2.TrackerKCF_create()),
            ('MOSSE (legacy)', lambda: cv2.legacy.TrackerMOSSE_create()),
            ('MOSSE', lambda: cv2.TrackerMOSSE_create()),
        ]
        
        for name, create_func in tracker_methods:
            try:
                tracker = create_func()
                if tracker is not None:
                    print(f"✅ 使用跟踪器: {name}")
                    break
            except (AttributeError, cv2.error):
                continue
        
        if tracker is None:
            self.log("❌ 无法创建跟踪器，请安装 opencv-contrib-python")
            self.log("   运行: pip install opencv-contrib-python")
            return
        
        tracker.init(self.current_frame, (x, y, w, h))
        
        # 创建跟踪对象
        obj_id = self.next_id
        self.next_id += 1
        
        history = deque(maxlen=50)  # 轨迹历史
        history.append((x + w//2, y + h//2))
        
        self.tracked_objects.append({
            'tracker': tracker,
            'label': label,
            'id': obj_id,
            'bbox': (x, y, w, h),
            'history': history,
            'lost_frames': 0,
            'active': True
        })
        
        self.log(f"✅ 添加跟踪 #{obj_id}: {label} at ({x},{y}) size {w}x{h}")
    
    def update_trackers(self, frame):
        """更新所有跟踪器"""
        active_objects = []
        
        for obj in self.tracked_objects:
            if not obj['active']:
                continue
            
            # 更新跟踪
            success, bbox = obj['tracker'].update(frame)
            
            if success:
                obj['bbox'] = tuple(map(int, bbox))
                x, y, w, h = obj['bbox']
                
                # 更新轨迹
                center = (x + w//2, y + h//2)
                obj['history'].append(center)
                
                obj['lost_frames'] = 0
                active_objects.append(obj)
            else:
                obj['lost_frames'] += 1
                
                # 连续丢失10帧则标记为失效
                if obj['lost_frames'] < 10:
                    active_objects.append(obj)
                else:
                    self.log(f"❌ 跟踪丢失 #{obj['id']}: {obj['label']}")
        
        self.tracked_objects = active_objects
    
    def get_current_annotations(self):
        """获取当前帧的标注数据"""
        annotations = []
        for obj in self.tracked_objects:
            if obj['active']:
                x, y, w, h = obj['bbox']
                annotations.append({
                    'id': int(obj['id']),
                    'label': str(obj['label']),
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'center': [int(x + w//2), int(y + h//2)]
                })
        return annotations
    
    def draw_annotations(self, frame):
        """绘制标注"""
        display = frame.copy()
        
        for obj in self.tracked_objects:
            if not obj['active']:
                continue
            
            x, y, w, h = obj['bbox']
            label = obj['label']
            obj_id = obj['id']
            
            # 颜色
            color = self.get_label_color(label)
            
            # 绘制边界框
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            if self.show_id:
                text = f"#{obj_id} {label}"
            else:
                text = label
            
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x, y - text_h - 10), 
                         (x + text_w + 10, y), color, -1)
            cv2.putText(display, text, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 绘制中心点
            center = (x + w//2, y + h//2)
            cv2.circle(display, center, 4, color, -1)
            
            # 绘制轨迹
            if self.show_trajectory and len(obj['history']) > 1:
                points = np.array(list(obj['history']), dtype=np.int32)
                cv2.polylines(display, [points], False, color, 2)
        
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
        
        # 跟踪数量
        cv2.putText(frame, f"Tracking: {len(self.tracked_objects)} objects", 
                   (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 录制状态
        if self.recording:
            rec_text = f"🔴 REC | Frame: {self.frame_count}"
            cv2.putText(frame, rec_text, (20, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press SPACE to start recording", 
                       (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 快捷键
        cv2.putText(frame, "SPACE:Record | T:Trajectory | C:Clear | Q:Quit",
                   (w - 550, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def start_recording(self):
        """开始录制"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 视频文件
        video_path = self.output_dir / f"annotated_video_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        size = (self.current_frame.shape[1], self.current_frame.shape[0])
        self.video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, size)
        
        # 重置标注数据
        self.frame_annotations = []
        self.frame_count = 0
        self.recording = True
        
        self.log(f"🎬 开始录制: {video_path}")
    
    def stop_recording(self):
        """停止录制"""
        if not self.recording:
            return
        
        self.recording = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # 保存标注数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"annotations_{timestamp}.json"
        
        data = {
            'timestamp': timestamp,
            'total_frames': len(self.frame_annotations),
            'labels': self.labels,
            'frames': self.frame_annotations
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 录制完成: {len(self.frame_annotations)} 帧")
        self.log(f"📊 标注数据已保存: {json_path}")
    
    def run(self, camera_id=0):
        """运行标注工具"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.log("❌ 无法打开摄像头")
            return
        
        window_name = "Video Auto Annotator"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.log("✅ 摄像头已打开")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.current_frame = frame.copy()
                
                # 更新跟踪
                if self.tracked_objects:
                    self.update_trackers(frame)
                
                # 录制
                if self.recording:
                    self.frame_count += 1
                    
                    # 记录当前帧的标注
                    frame_data = {
                        'frame': self.frame_count,
                        'annotations': self.get_current_annotations()
                    }
                    self.frame_annotations.append(frame_data)
                
                # 绘制
                display = self.draw_annotations(frame)
                
                # 写入视频
                if self.recording and self.video_writer:
                    self.video_writer.write(display)
                
                cv2.imshow(window_name, display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if self.recording:
                        self.stop_recording()
                    else:
                        if self.tracked_objects:
                            self.start_recording()
                        else:
                            self.log("⚠️  请先添加跟踪对象")
                elif key == ord('t'):
                    self.show_trajectory = not self.show_trajectory
                    status = "开启" if self.show_trajectory else "关闭"
                    self.log(f"🎨 轨迹显示: {status}")
                elif key == ord('i'):
                    self.show_id = not self.show_id
                    status = "开启" if self.show_id else "关闭"
                    self.log(f"🏷️  ID显示: {status}")
                elif key == ord('c'):
                    self.tracked_objects = []
                    self.log("🗑️  已清除所有跟踪")
                elif key == ord('d'):
                    if self.tracked_objects:
                        removed = self.tracked_objects.pop()
                        self.log(f"↩️  已删除 #{removed['id']}: {removed['label']}")
                elif key == ord('s'):
                    # 保存当前帧标注
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = self.output_dir / f"frame_{timestamp}.jpg"
                    cv2.imwrite(str(img_path), display)
                    
                    json_path = self.output_dir / f"frame_{timestamp}.json"
                    frame_data = {
                        'timestamp': timestamp,
                        'annotations': self.get_current_annotations()
                    }
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(frame_data, f, indent=2, ensure_ascii=False)
                    
                    self.log(f"📸 已保存: {img_path}")
                elif key == ord('h'):
                    self.print_help()
                elif ord('1') <= key <= ord('9'):
                    idx = key - ord('1')
                    if idx < len(self.labels):
                        self.current_label_idx = idx
                        self.log(f"🏷️  切换到: {self.labels[idx]}")
        
        except KeyboardInterrupt:
            self.log("⚠️  用户中断")
        finally:
            if self.recording:
                self.stop_recording()
            
            cap.release()
            cv2.destroyAllWindows()
            self.log_fp.close()
            self.log("👋 已退出")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='视频自动标注工具')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--output', type=str, default='data/video_annotations',
                       help='输出目录')
    args = parser.parse_args()
    
    annotator = VideoAutoAnnotator(args.output)
    annotator.run(args.camera)

if __name__ == "__main__":
    main()
