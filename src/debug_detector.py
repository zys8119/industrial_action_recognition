#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试版检测器 - 显示详细信息
"""

import cv2
import numpy as np
from pathlib import Path
import pickle

def test_detection():
    """测试检测功能"""
    
    print("🔍 检测调试工具\n")
    
    # 1. 加载模型
    print("1️⃣ 加载模型...")
    model_path = "models/video_trained_model.pkl"
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    templates = model_data['templates']
    labels = model_data['labels']
    
    print(f"   ✅ 类别: {labels}")
    print(f"   ✅ 总样本: {model_data['total_samples']}")
    
    for label, tmps in templates.items():
        print(f"   📊 {label}: {len(tmps)} 个模板")
        if tmps:
            print(f"      - 第1个模板: {tmps[0]['descriptors'].shape}")
            print(f"      - 尺寸: {tmps[0]['size']}")
    
    # 2. 打开摄像头
    print("\n2️⃣ 打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("   ❌ 无法打开摄像头")
        return
    
    print("   ✅ 摄像头已打开")
    
    # 3. 读取一帧
    print("\n3️⃣ 读取画面...")
    ret, frame = cap.read()
    
    if not ret:
        print("   ❌ 无法读取画面")
        cap.release()
        return
    
    print(f"   ✅ 画面尺寸: {frame.shape}")
    
    # 4. 提取特征
    print("\n4️⃣ 提取画面特征...")
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(gray, None)
    
    if des_frame is None:
        print("   ❌ 无法提取特征")
        cap.release()
        return
    
    print(f"   ✅ 特征点数: {len(kp_frame)}")
    print(f"   ✅ 描述符: {des_frame.shape}")
    
    # 5. 特征匹配
    print("\n5️⃣ 特征匹配测试...")
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    for label, tmps in templates.items():
        print(f"\n   测试类别: {label}")
        
        best_match_count = 0
        
        for i, template in enumerate(tmps[:3]):  # 只测试前3个
            try:
                matches = bf_matcher.match(template['descriptors'], des_frame)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = matches[:min(50, len(matches))]
                
                if len(good_matches) > best_match_count:
                    best_match_count = len(good_matches)
                
                print(f"      模板 {i+1}: {len(good_matches)} 个匹配点")
                
                if len(good_matches) >= 10:
                    avg_distance = np.mean([m.distance for m in good_matches])
                    confidence = max(0, 1.0 - avg_distance / 200.0)
                    print(f"         平均距离: {avg_distance:.2f}")
                    print(f"         置信度: {confidence:.2%}")
                    
                    if confidence > 0.3:
                        print(f"         ✅ 可以检测！")
                    else:
                        print(f"         ⚠️  置信度太低")
            
            except Exception as e:
                print(f"      模板 {i+1}: 匹配失败 - {e}")
        
        print(f"   最佳匹配: {best_match_count} 个点")
        
        if best_match_count < 10:
            print(f"   ❌ 匹配点太少，无法检测")
            print(f"   💡 建议:")
            print(f"      - 将物体放在摄像头前")
            print(f"      - 确保物体清晰可见")
            print(f"      - 使用与训练时相似的角度和距离")
        else:
            print(f"   ✅ 匹配点足够，应该能检测")
    
    # 6. 可视化特征点
    print("\n6️⃣ 保存特征点可视化...")
    vis_frame = cv2.drawKeypoints(frame, kp_frame, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("debug_features.jpg", vis_frame)
    print("   ✅ 已保存: debug_features.jpg")
    
    cap.release()
    
    print("\n" + "="*60)
    print("调试完成！")
    print("="*60)
    
    print("\n💡 下一步:")
    print("   1. 查看 debug_features.jpg，确认画面中有足够的特征点")
    print("   2. 如果匹配点太少，尝试:")
    print("      - 将物体放在摄像头前")
    print("      - 改善光照")
    print("      - 使用纹理丰富的物体")
    print("   3. 如果匹配点足够但检测不到，可能是阈值问题")

if __name__ == "__main__":
    test_detection()
