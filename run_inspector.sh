#!/bin/bash
# 视觉检查工具启动脚本

echo "🔍 视觉检查工具"
echo ""
echo "用法示例："
echo ""
echo "1. 摄像头实时检测："
echo "   ./run_inspector.sh --template reference.jpg"
echo ""
echo "2. 检查单张图片："
echo "   ./run_inspector.sh --template reference.jpg --image test.jpg"
echo ""
echo "3. 指定模板名称："
echo "   ./run_inspector.sh --template bolt.jpg --name '螺栓'"
echo ""
echo "快捷键："
echo "   Q - 退出"
echo "   S - 保存当前帧"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
python src/visual_inspector.py "$@"
