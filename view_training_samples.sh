#!/bin/bash
# 查看训练样本预览

echo "🎨 训练样本预览"
echo ""

cd "$(dirname "$0")"

# 检查目录是否存在
if [ ! -d "training_samples_preview" ]; then
    echo "⚠️  训练样本预览目录不存在"
    echo ""
    echo "💡 请先生成预览："
    echo "   source venv/bin/activate"
    echo "   python src/visualize_training.py"
    exit 1
fi

# 统计文件数量
frame_count=$(ls training_samples_preview/frame_*.jpg 2>/dev/null | wc -l)
sample_count=$(ls training_samples_preview/sample_*.jpg 2>/dev/null | wc -l)

echo "📊 统计信息："
echo "   完整帧: $frame_count 张"
echo "   物体样本: $sample_count 张"
echo ""

if [ $frame_count -eq 0 ] && [ $sample_count -eq 0 ]; then
    echo "❌ 目录为空"
    exit 1
fi

echo "🖼️  正在打开预览目录..."
echo ""

# 根据操作系统打开目录
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open training_samples_preview/
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open &> /dev/null; then
        xdg-open training_samples_preview/
    elif command -v nautilus &> /dev/null; then
        nautilus training_samples_preview/
    else
        echo "请手动打开目录: training_samples_preview/"
    fi
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    explorer training_samples_preview
else
    echo "请手动打开目录: training_samples_preview/"
fi

echo "✅ 已打开训练样本预览目录"
echo ""
echo "💡 检查内容："
echo "   1. frame_XXX.jpg - 完整帧（带绿色标注框）"
echo "   2. sample_XXX_yazi.jpg - 提取的物体图像"
echo ""
echo "🔍 确认："
echo "   ✅ 绿色框是否准确框住了目标物体？"
echo "   ✅ 提取的物体图像是否清晰？"
echo "   ✅ 是否标注了正确的物体？"
echo ""
echo "如果标注有问题，需要重新录制视频！"
