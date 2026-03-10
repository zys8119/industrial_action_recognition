#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建示例参考图片 - 用于测试视觉检查工具
"""

import cv2
import numpy as np
from pathlib import Path

def create_sample_reference():
    """创建一个示例参考图片（简单的几何图形）"""
    
    # 创建白色背景
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # 绘制一个蓝色圆形（模拟螺栓头）
    cv2.circle(img, (200, 200), 80, (255, 100, 0), -1)
    cv2.circle(img, (200, 200), 80, (200, 80, 0), 3)
    
    # 绘制中心孔
    cv2.circle(img, (200, 200), 20, (255, 255, 255), -1)
    cv2.circle(img, (200, 200), 20, (200, 80, 0), 2)
    
    # 添加一些纹理（螺纹）
    for i in range(6):
        angle = i * 60
        x1 = int(200 + 50 * np.cos(np.radians(angle)))
        y1 = int(200 + 50 * np.sin(np.radians(angle)))
        x2 = int(200 + 70 * np.cos(np.radians(angle)))
        y2 = int(200 + 70 * np.sin(np.radians(angle)))
        cv2.line(img, (x1, y1), (x2, y2), (150, 60, 0), 2)
    
    # 添加文字标签
    cv2.putText(img, "M8 Bolt", (150, 350), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img

def main():
    # 创建示例目录
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 创建参考图片
    reference = create_sample_reference()
    ref_path = examples_dir / "bolt_reference.jpg"
    cv2.imwrite(str(ref_path), reference)
    
    print(f"✅ 示例参考图片已创建: {ref_path}")
    print(f"\n使用方法:")
    print(f"  ./run_inspector.sh --template {ref_path}")
    print(f"\n或者使用你自己的参考图片:")
    print(f"  ./run_inspector.sh --template your_image.jpg")

if __name__ == "__main__":
    main()
