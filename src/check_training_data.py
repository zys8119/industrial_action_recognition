#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查训练数据质量
"""

import cv2
import json
from pathlib import Path

def check_training_data():
    """检查训练数据"""
    
    print("🔍 检查训练数据质量\n")
    
    annotations_dir = Path("data/video_annotations")
    
    # 查找标注文件
    json_files = list(annotations_dir.glob("annotations_*.json"))
    
    if not json_files:
        print("❌ 没有找到标注文件")
        return
    
    print(f"📂 找到 {len(json_files)} 个标注文件\n")
    
    for json_file in json_files:
        print(f"{'='*60}")
        print(f"文件: {json_file.name}")
        print(f"{'='*60}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"总帧数: {data['total_frames']}")
        print(f"类别: {data['labels']}")
        
        # 统计每个类别的标注数量
        label_counts = {}
        total_annotations = 0
        
        for frame_data in data['frames']:
            for ann in frame_data['annotations']:
                label = ann['label']
                label_counts[label] = label_counts.get(label, 0) + 1
                total_annotations += 1
        
        print(f"\n标注统计:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} 次")
        
        print(f"总标注数: {total_annotations}")
        
        # 检查标注的尺寸分布
        if data['frames']:
            sizes = []
            for frame_data in data['frames']:
                for ann in frame_data['annotations']:
                    bbox = ann['bbox']
                    w, h = bbox[2], bbox[3]
                    sizes.append((w, h))
            
            if sizes:
                import numpy as np
                widths = [s[0] for s in sizes]
                heights = [s[1] for s in sizes]
                
                print(f"\n尺寸分布:")
                print(f"  宽度: {np.min(widths):.0f} - {np.max(widths):.0f} (平均: {np.mean(widths):.0f})")
                print(f"  高度: {np.min(heights):.0f} - {np.max(heights):.0f} (平均: {np.mean(heights):.0f})")
        
        print()
    
    print("\n" + "="*60)
    print("💡 建议:")
    print("="*60)
    print("1. 确认标注的类别名称是否正确")
    print("2. 确认标注的是你想要识别的物体")
    print("3. 如果标注错误，需要重新录制视频")
    print("4. 每个类别建议至少 50 个标注")

if __name__ == "__main__":
    check_training_data()
