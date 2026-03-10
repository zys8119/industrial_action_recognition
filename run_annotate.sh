#!/bin/bash
# 启动实时标注工具

echo "🎨 启动图像标注工具..."
echo ""
echo "⌨️  快捷键:"
echo "   鼠标拖动    - 绘制边界框"
echo "   数字键 1-9  - 切换类别"
echo "   SPACE       - 保存当前帧"
echo "   C           - 清除标注"
echo "   U           - 撤销"
echo "   S           - 导出JSON"
echo "   H           - 帮助"
echo "   Q           - 退出"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
python src/annotate_tool.py "$@"
