#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于视频标注数据训练检测器
将视频标注数据转换为训练样本，并训练简单的检测模型
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pickle

class VideoAnnotationTrainer:
    """基于视频标注数据的训练器"""
    
    def __init__(self, annotations_dir="data/video_annotations"):
        self.annotations_dir = Path(annotations_dir)
        self.templates = {}  # {label: [template_data, ...]}
        
        # 特征提取器
        self.sift = cv2.SIFT_create()
        
        print("🎓 视频标注数据训练器")
    
    def load_video_annotations(self):
        """加载视频标注数据"""
        json_files = list(self.annotations_dir.glob("annotations_*.json"))
        
        if not json_files:
            print(f"❌ 未找到标注文件")
            print(f"💡 请先运行视频标注工具并录制视频")
            return False
        
        print(f"📂 找到 {len(json_files)} 个标注文件")
        
        # 获取所有视频文件
        video_files = list(self.annotations_dir.glob("annotated_video_*.mp4"))
        
        if not video_files:
            print(f"❌ 未找到视频文件")
            return False
        
        print(f"📹 找到 {len(video_files)} 个视频文件")
        
        total_samples = 0
        
        for json_file in json_files:
            print(f"\n处理: {json_file.name}")
            
            # 读取标注数据
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 尝试多种方式匹配视频文件
            video_file = None
            
            # 方法1: 精确匹配（替换文件名）
            expected_video = json_file.parent / json_file.name.replace('annotations_', 'annotated_video_').replace('.json', '.mp4')
            if expected_video.exists():
                video_file = expected_video
            else:
                # 方法2: 时间戳匹配（找最接近的视频）
                json_timestamp = json_file.stem.split('_')[-1]  # 提取时间戳
                
                best_match = None
                min_diff = float('inf')
                
                for vf in video_files:
                    video_timestamp = vf.stem.split('_')[-1]
                    # 计算时间戳差异
                    try:
                        diff = abs(int(json_timestamp) - int(video_timestamp))
                        if diff < min_diff:
                            min_diff = diff
                            best_match = vf
                    except:
                        continue
                
                # 如果时间差小于60秒，认为是匹配的
                if best_match and min_diff < 100:  # 时间戳格式 HHMMSS，100 = 1分钟
                    video_file = best_match
                    print(f"   匹配到视频: {video_file.name} (时间差: {min_diff}秒)")
            
            if not video_file or not video_file.exists():
                print(f"⚠️  未找到匹配的视频文件")
                continue
            
            # 打开视频
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"❌ 无法打开视频")
                continue
            
            # 处理每一帧
            frame_idx = 0
            samples_from_video = 0
            
            for frame_data in data['frames']:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # 每5帧采样一次（减少冗余）
                if frame_idx % 5 != 0:
                    continue
                
                # 提取每个标注对象
                for ann in frame_data['annotations']:
                    label = ann['label']
                    x, y, w, h = ann['bbox']
                    
                    # 提取ROI
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0 or w < 20 or h < 20:
                        continue
                    
                    # 提取特征
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    keypoints, descriptors = self.sift.detectAndCompute(gray_roi, None)
                    
                    if descriptors is None or len(keypoints) < 5:
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
                        'source': f"{json_file.name}:frame{frame_idx}"
                    })
                    
                    samples_from_video += 1
                    total_samples += 1
            
            cap.release()
            print(f"✅ 提取了 {samples_from_video} 个样本")
        
        print(f"\n📊 训练数据统计:")
        for label, templates in self.templates.items():
            print(f"   {label}: {len(templates)} 个样本")
        print(f"   总计: {total_samples} 个样本")
        
        return total_samples > 0
    
    def save_model(self, output_path="models/video_trained_model.pkl"):
        """保存训练好的模型"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换模板数据（只保存描述符，不保存 KeyPoint 对象）
        serializable_templates = {}
        for label, templates in self.templates.items():
            serializable_templates[label] = []
            for template in templates:
                serializable_templates[label].append({
                    'descriptors': template['descriptors'],
                    'size': template['size'],
                    'source': template['source']
                })
        
        # 保存模板数据
        model_data = {
            'templates': serializable_templates,
            'labels': list(self.templates.keys()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': sum(len(t) for t in self.templates.values())
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 模型已保存: {output_path}")
        print(f"📊 包含 {model_data['total_samples']} 个训练样本")
        
        return output_path
    
    def train(self):
        """训练模型"""
        print("🎓 开始训练...\n")
        
        # 加载标注数据
        if not self.load_video_annotations():
            return None
        
        # 保存模型
        model_path = self.save_model()
        
        print("\n✅ 训练完成！")
        print(f"\n💡 下一步:")
        print(f"   python src/detect_from_video_model.py --model {model_path}")
        
        return model_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='基于视频标注数据训练检测器')
    parser.add_argument('--annotations', type=str, 
                       default='data/video_annotations',
                       help='视频标注数据目录')
    parser.add_argument('--output', type=str,
                       default='models/video_trained_model.pkl',
                       help='输出模型路径')
    args = parser.parse_args()
    
    trainer = VideoAnnotationTrainer(args.annotations)
    trainer.train()

if __name__ == "__main__":
    main()
