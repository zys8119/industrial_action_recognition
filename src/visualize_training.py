#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化训练样本
"""

import cv2
import json
from pathlib import Path
import numpy as np

def visualize_training_samples():
    """可视化训练样本"""
    
    print("🎨 可视化训练样本\n")
    
    annotations_dir = Path("data/video_annotations")
    
    # 查找标注文件和视频
    json_files = list(annotations_dir.glob("annotations_*.json"))
    
    if not json_files:
        print("❌ 没有找到标注文件")
        return
    
    # 使用第一个标注文件
    json_file = json_files[0]
    print(f"使用标注文件: {json_file.name}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 找到对应的视频
    video_files = list(annotations_dir.glob("annotated_video_*.mp4"))
    
    if not video_files:
        print("❌ 没有找到视频文件")
        return
    
    # 智能匹配视频
    json_timestamp = json_file.stem.split('_')[-1]
    best_match = None
    min_diff = float('inf')
    
    for vf in video_files:
        video_timestamp = vf.stem.split('_')[-1]
        try:
            diff = abs(int(json_timestamp) - int(video_timestamp))
            if diff < min_diff:
                min_diff = diff
                best_match = vf
        except:
            continue
    
    if not best_match:
        print("❌ 无法匹配视频文件")
        return
    
    print(f"使用视频文件: {best_match.name}\n")
    
    # 打开视频
    cap = cv2.VideoCapture(str(best_match))
    
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return
    
    # 创建输出目录
    output_dir = Path("training_samples_preview")
    output_dir.mkdir(exist_ok=True)
    
    print("📸 提取训练样本...")
    
    # 每隔 N 帧提取一个样本
    sample_interval = max(1, len(data['frames']) // 20)  # 最多提取20个样本
    
    samples_saved = 0
    
    for frame_idx, frame_data in enumerate(data['frames']):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 只保存部分样本
        if frame_idx % sample_interval != 0:
            continue
        
        # 绘制标注
        for ann in frame_data['annotations']:
            x, y, w, h = ann['bbox']
            label = ann['label']
            
            # 绘制框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 提取ROI
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                roi_path = output_dir / f"sample_{samples_saved:03d}_{label}.jpg"
                cv2.imwrite(str(roi_path), roi)
        
        # 保存完整帧
        frame_path = output_dir / f"frame_{samples_saved:03d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        samples_saved += 1
        
        if samples_saved >= 20:
            break
    
    cap.release()
    
    print(f"\n✅ 已保存 {samples_saved} 个样本到: {output_dir}")
    print(f"\n💡 请查看 {output_dir} 目录:")
    print(f"   - frame_XXX.jpg: 完整帧（带标注框）")
    print(f"   - sample_XXX_yazi.jpg: 提取的物体图像")
    print(f"\n🔍 检查这些图像:")
    print(f"   1. 标注框是否准确框住了你想要的物体？")
    print(f"   2. 提取的物体图像是否清晰？")
    print(f"   3. 是否标注了错误的物体？")
    print(f"\n如果标注有问题，需要重新录制视频！")

if __name__ == "__main__":
    visualize_training_samples()
