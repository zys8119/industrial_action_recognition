#!/bin/bash
# 精确检测器 - 更准确的位置和大小

echo "🎯 精确检测器"
echo ""
echo "✨ 改进："
echo "   ✅ 使用凸包计算边界"
echo "   ✅ 保持物体长宽比"
echo "   ✅ 加权平滑算法"
echo "   ✅ 显示精确尺寸"
echo ""
echo "💡 特点："
echo "   - 边界框更贴合物体"
echo "   - 尺寸更准确"
echo "   - 位置更稳定"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
python src/accurate_detector.py --model models/video_trained_model.pkl "$@"
