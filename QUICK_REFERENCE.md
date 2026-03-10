# 快速参考 - 所有工具和脚本

## 🚀 快速启动

```bash
cd ~/Desktop/industrial_action_recognition

# 工具菜单（推荐）
./tools.sh
```

## 📋 所有脚本列表

### 数据采集

| 脚本 | 功能 | 使用场景 |
|------|------|---------|
| `./run_video_annotator.sh` | 视频自动标注 | 录制视频并自动跟踪标注 ✅ |
| `./run_annotate.sh` | 静态图像标注 | 逐帧手动标注 |
| `./run_smart_annotator.sh` | 智能标注 | 带边缘吸附的标注 |
| `./run_tracking.sh` | 动态跟踪 | 手动标记后自动跟踪 |

### 模型训练

| 脚本 | 功能 |
|------|------|
| `python src/train_from_video.py` | 从视频标注训练模型 |
| `./view_training_samples.sh` | 查看训练样本预览 ✅ |
| `python src/check_training_data.py` | 检查训练数据统计 |
| `python src/visualize_training.py` | 生成训练样本预览 |

### 物体检测

| 脚本 | 特点 | 推荐场景 |
|------|------|---------|
| `./run_simple_detector.sh` | 快速检测 | 实时性要求高 |
| `./run_accurate_detector.sh` | 精确检测 | 需要准确位置和尺寸 ✅ |
| `./run_detector.sh` | 基于标注 | 使用静态标注数据 |
| `./run_camera.sh` | 动作识别 | 演示模式 |

### 调试工具

| 脚本 | 功能 |
|------|------|
| `python src/debug_detector.py` | 调试检测器 |
| `python src/prepare_data.py` | 准备训练数据 |

### 其他工具

| 脚本 | 功能 |
|------|------|
| `./run_inspector.sh` | 视觉检查（零训练） |
| `python src/view_log.py` | 查看日志 |

## 🎯 完整工作流程

### 方案 1：视频标注（推荐）

```bash
# 1. 录制并标注
./run_video_annotator.sh
# 操作：框选物体 → 空格开始录制 → 空格停止

# 2. 查看训练样本
./view_training_samples.sh
# 确认标注是否正确

# 3. 训练模型
source venv/bin/activate
python src/train_from_video.py

# 4. 检测
./run_accurate_detector.sh
```

### 方案 2：静态标注

```bash
# 1. 标注图像
./run_smart_annotator.sh
# 操作：框选物体 → 空格保存

# 2. 检测
./run_detector.sh
```

## ⌨️ 常用快捷键

### 标注工具
- **鼠标拖动** - 绘制框
- **数字键 1-9** - 切换类别
- **空格** - 保存/开始录制
- **C** - 清除
- **Q** - 退出

### 检测工具
- **Q** - 退出
- **S** - 截图
- **R** - 重置

## 📁 重要目录

```
industrial_action_recognition/
├── data/
│   ├── video_annotations/      # 视频标注数据
│   ├── smart_annotations/      # 智能标注数据
│   └── tracking/               # 跟踪数据
├── models/
│   └── video_trained_model.pkl # 训练好的模型
├── training_samples_preview/   # 训练样本预览
├── logs/                       # 日志文件
└── screenshots/                # 截图
```

## 🔧 常用命令

### 查看训练数据
```bash
source venv/bin/activate
python src/check_training_data.py
```

### 生成训练样本预览
```bash
source venv/bin/activate
python src/visualize_training.py
./view_training_samples.sh
```

### 调试检测
```bash
source venv/bin/activate
python src/debug_detector.py
```

### 查看日志
```bash
source venv/bin/activate
python src/view_log.py --tail 50
python src/view_log.py --analyze
```

## 💡 快速解决问题

### 检测不到物体？
```bash
# 1. 检查训练数据
./view_training_samples.sh

# 2. 调试检测
source venv/bin/activate
python src/debug_detector.py

# 3. 重新训练
python src/train_from_video.py
```

### 位置不准确？
```bash
# 使用精确检测器
./run_accurate_detector.sh
```

### 标注错误？
```bash
# 1. 删除旧数据
rm -rf data/video_annotations/*.mp4
rm -rf data/video_annotations/*.json

# 2. 重新标注
./run_video_annotator.sh

# 3. 重新训练
source venv/bin/activate
python src/train_from_video.py
```

## 📚 文档

- `README.md` - 项目说明
- `QUICKSTART.md` - 快速开始
- `COMPLETE_WORKFLOW.md` - 完整工作流程
- `VIDEO_ANNOTATION_GUIDE.md` - 视频标注指南
- `DETECTION_TROUBLESHOOTING.md` - 检测问题排查
- `ACCURACY_IMPROVEMENT_GUIDE.md` - 准确性改进

## 🎓 学习路径

1. **入门** - 阅读 `QUICKSTART.md`
2. **标注** - 运行 `./run_video_annotator.sh`
3. **训练** - 运行 `python src/train_from_video.py`
4. **检测** - 运行 `./run_accurate_detector.sh`
5. **优化** - 阅读 `ACCURACY_IMPROVEMENT_GUIDE.md`

## 🆘 获取帮助

```bash
# 查看工具菜单
./tools.sh

# 查看脚本帮助
./run_video_annotator.sh --help
python src/train_from_video.py --help
```

现在你有了完整的工具集！使用 `./tools.sh` 可以快速访问所有功能。
