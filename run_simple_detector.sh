#!/bin/bash
# 简化版检测器启动脚本

echo "🔍 简化版物体检测器"
echo ""
echo "✨ 特点："
echo "   ✅ 更简单的匹配算法"
echo "   ✅ 更快的检测速度"
echo "   ✅ 更好的可视化"
echo ""
echo "💡 使用方法："
echo "   1. 将物体放在摄像头前"
echo "   2. 保持物体清晰可见"
echo "   3. 系统会自动检测并标注"
echo ""
echo "⌨️  快捷键："
echo "   Q - 退出"
echo "   S - 截图"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
python src/simple_detector.py --model models/video_trained_model.pkl "$@"
