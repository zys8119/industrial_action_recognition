#!/bin/bash
# 智能标注工具启动脚本

echo "🎨 智能标注工具"
echo ""
echo "✨ 智能辅助功能："
echo "   ✅ 边缘吸附 - 自动对齐物体边缘"
echo "   ✅ 自动调整 - 智能优化边界框"
echo "   ✅ 智能建议 - 自动检测候选区域"
echo ""
echo "⌨️  快捷键："
echo "   鼠标拖动    - 绘制标注框"
echo "   E           - 切换边缘吸附"
echo "   A           - 切换自动调整"
echo "   S           - 切换智能建议"
echo "   SPACE       - 保存标注"
echo "   C           - 清除所有"
echo "   U           - 撤销"
echo "   Q           - 退出"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
python src/smart_annotator.py "$@"
