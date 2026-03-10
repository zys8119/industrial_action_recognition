#!/bin/bash
# 启动动态跟踪工具

echo "🎯 启动图像标注与动态跟踪工具..."
echo ""
echo "⌨️  快捷键:"
echo "   鼠标拖动    - 标记物体（自动跟踪）"
echo "   数字键 1-9  - 切换类别"
echo "   SPACE       - 暂停/继续"
echo "   C           - 清除所有跟踪"
echo "   D           - 删除最后一个"
echo "   T           - 切换轨迹显示"
echo "   L           - 切换标签显示"
echo "   S           - 保存当前帧"
echo "   R           - 切换跟踪器"
echo "   H           - 帮助"
echo "   Q           - 退出"
echo ""
echo "🎯 跟踪器类型:"
echo "   CSRT  - 精确但慢（默认）"
echo "   KCF   - 平衡"
echo "   MOSSE - 快速但不精确"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
python src/tracking_tool.py "$@"
