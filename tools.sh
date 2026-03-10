#!/bin/bash
# 工业物体识别工具集

echo "🎯 工业物体识别工具集"
echo ""
echo "请选择要运行的工具："
echo ""
echo "📹 数据采集："
echo "  1) 视频自动标注 - 录制并自动跟踪标注"
echo "  2) 静态图像标注 - 逐帧手动标注"
echo "  3) 智能标注工具 - 带边缘吸附的标注"
echo ""
echo "🎓 模型训练："
echo "  4) 训练模型 - 从视频标注数据训练"
echo "  5) 查看训练样本 - 检查标注质量"
echo "  6) 检查训练数据 - 统计信息"
echo ""
echo "🔍 物体检测："
echo "  7) 简单检测器 - 快速检测"
echo "  8) 精确检测器 - 准确的位置和尺寸"
echo "  9) 基于标注的检测 - 使用静态标注"
echo ""
echo "🛠️  调试工具："
echo "  10) 调试检测器 - 查看详细匹配信息"
echo "  11) 可视化训练数据 - 生成预览图像"
echo ""
echo "  0) 退出"
echo ""
read -p "请输入选项 (0-11): " choice

case $choice in
    1)
        echo ""
        echo "🎬 启动视频自动标注工具..."
        ./run_video_annotator.sh
        ;;
    2)
        echo ""
        echo "🎨 启动静态图像标注工具..."
        ./run_annotate.sh
        ;;
    3)
        echo ""
        echo "✨ 启动智能标注工具..."
        ./run_smart_annotator.sh
        ;;
    4)
        echo ""
        echo "🎓 开始训练模型..."
        source venv/bin/activate
        python src/train_from_video.py
        ;;
    5)
        echo ""
        echo "🖼️  查看训练样本..."
        ./view_training_samples.sh
        ;;
    6)
        echo ""
        echo "📊 检查训练数据..."
        source venv/bin/activate
        python src/check_training_data.py
        ;;
    7)
        echo ""
        echo "🔍 启动简单检测器..."
        ./run_simple_detector.sh
        ;;
    8)
        echo ""
        echo "🎯 启动精确检测器..."
        ./run_accurate_detector.sh
        ;;
    9)
        echo ""
        echo "📋 启动基于标注的检测器..."
        ./run_detector.sh
        ;;
    10)
        echo ""
        echo "🐛 启动调试检测器..."
        source venv/bin/activate
        python src/debug_detector.py
        ;;
    11)
        echo ""
        echo "🎨 生成训练数据可视化..."
        source venv/bin/activate
        python src/visualize_training.py
        ./view_training_samples.sh
        ;;
    0)
        echo ""
        echo "👋 再见！"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ 无效选项"
        exit 1
        ;;
esac
