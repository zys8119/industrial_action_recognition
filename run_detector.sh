#!/bin/bash
# 基于标注数据的检测器

echo "🎯 基于标注数据的物体检测"
echo ""
echo "📋 工作流程："
echo "   1. 先运行标注工具标记物体"
echo "   2. 保存标注数据（按空格）"
echo "   3. 运行此脚本自动识别"
echo ""
echo "💡 快捷键："
echo "   Q - 退出"
echo "   S - 截图保存"
echo "   R - 重置平滑"
echo ""
echo "✨ 已启用平滑算法，减少抖动"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
python src/detect_from_annotations.py "$@"
