#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动准备训练数据
扫描 data/videos/ 目录，自动生成标注文件
"""

import os
from pathlib import Path

def prepare_data():
    """自动生成训练和验证数据列表"""
    
    base_dir = Path(__file__).parent.parent
    videos_dir = base_dir / "data" / "videos"
    annotations_dir = base_dir / "data" / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    
    # 读取类别列表
    label_file = base_dir / "configs" / "label_list.txt"
    if not label_file.exists():
        print("❌ 未找到 configs/label_list.txt，请先创建类别文件")
        return
    
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    print(f"📋 发现 {len(labels)} 个动作类别: {labels}")
    
    # 扫描视频文件
    train_list = []
    val_list = []
    
    for label_id, label_name in enumerate(labels):
        label_dir = videos_dir / label_name
        
        if not label_dir.exists():
            print(f"⚠️  未找到目录: {label_dir}，跳过")
            label_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ 已创建目录: {label_dir}")
            continue
        
        # 查找视频文件
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
            video_files.extend(label_dir.glob(ext))
        
        if not video_files:
            print(f"⚠️  {label_name} 目录下没有视频文件")
            continue
        
        print(f"📹 {label_name}: 找到 {len(video_files)} 个视频")
        
        # 80% 训练，20% 验证
        # 如果样本太少（<5个），全部用于训练
        if len(video_files) < 5:
            split_idx = len(video_files)
            print(f"   ⚠️  样本较少，全部用于训练")
        else:
            split_idx = max(1, int(len(video_files) * 0.8))
        
        for i, video_file in enumerate(video_files):
            # 相对于 data 目录的路径
            rel_path = video_file.relative_to(base_dir / "data")
            line = f"{rel_path} {label_id}\n"
            
            if i < split_idx:
                train_list.append(line)
            else:
                val_list.append(line)
    
    # 写入标注文件
    train_file = annotations_dir / "train_list.txt"
    val_file = annotations_dir / "val_list.txt"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_list)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_list)
    
    print(f"\n✅ 数据准备完成！")
    print(f"   训练样本: {len(train_list)}")
    print(f"   验证样本: {len(val_list)}")
    print(f"   标注文件: {train_file}")
    
    if len(train_list) == 0:
        print(f"\n❌ 没有训练数据！")
        print(f"💡 请将视频文件放入 data/videos/ 对应的类别文件夹")
        print(f"   例如: data/videos/搬运箱子/video1.mp4")
        print(f"\n📋 当前类别: {', '.join(labels)}")
        return
    
    if len(train_list) < 10:
        print(f"\n⚠️  训练样本较少（建议每类至少 20 个视频）")
        print(f"💡 当前可以训练，但效果可能不佳")
    
    print(f"\n💡 下一步: python src/train.py")

if __name__ == "__main__":
    prepare_data()
