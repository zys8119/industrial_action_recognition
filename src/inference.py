#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频文件推理脚本
"""

import cv2
import argparse
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent))
from camera_demo import ActionRecognizer

def inference_video(video_path, model_path=None, output_path=None):
    """对视频文件进行推理"""
    
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 初始化识别器
    recognizer = ActionRecognizer(model_path)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 视频信息:")
    print(f"   分辨率: {width}x{height}")
    print(f"   帧率: {fps} FPS")
    print(f"   总帧数: {total_frames}")
    
    # 输出视频
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 输出视频: {output_path}")
    
    frame_count = 0
    action_history = []
    
    print("\n🎬 开始处理...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 预测动作
        action, confidence = recognizer.predict(frame)
        
        if action:
            action_history.append((frame_count, action, confidence))
        
        # 绘制结果
        frame = recognizer.draw_results(frame, action, confidence)
        
        # 写入输出视频
        if writer:
            writer.write(frame)
        
        # 显示进度
        if frame_count % 30 == 0:
            progress = frame_count / total_frames * 100
            print(f"   进度: {progress:.1f}% ({frame_count}/{total_frames})")
    
    cap.release()
    if writer:
        writer.release()
    
    print(f"\n✅ 处理完成！")
    print(f"📊 识别结果统计:")
    
    # 统计各动作出现次数
    action_counts = {}
    for _, action, _ in action_history:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(action_history) * 100 if action_history else 0
        print(f"   {action}: {count} 次 ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='视频文件动作识别')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--output', type=str, help='输出视频路径（可选）')
    args = parser.parse_args()
    
    inference_video(args.video, args.model, args.output)

if __name__ == "__main__":
    main()
