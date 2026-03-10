#!/bin/bash
# 快速测试摄像头 Demo（带日志）

echo "🎥 启动工业动作识别摄像头 Demo..."
echo ""
echo "💡 提示："
echo "   - 按 'q' 退出"
echo "   - 按 'r' 重置缓冲区"
echo "   - 按 's' 截图保存"
echo ""
echo "📝 日志文件: logs/recognition.log"
echo "📸 截图目录: screenshots/"
echo ""
echo "⚠️  当前为演示模式（模拟检测）"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
python src/camera_demo.py "$@"
