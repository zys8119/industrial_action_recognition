# 快速开始指南

## 1️⃣ 准备训练数据

### 创建动作类别文件夹

在 `data/videos/` 下为每个动作创建文件夹：

```bash
cd ~/Desktop/industrial_action_recognition/data/videos

# 创建三个示例类别
mkdir -p welding assembly inspection
```

### 放入视频文件

将你的训练视频按类别放入对应文件夹：

```
data/videos/
├── welding/          # 焊接动作视频
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── assembly/         # 装配动作视频
│   ├── video1.mp4
│   └── video2.mp4
└── inspection/       # 检查动作视频
    └── video1.mp4
```

**建议：**
- 每个类别至少 20-50 个视频
- 视频时长 3-10 秒
- 清晰展示完整动作
- 不同角度、光照条件

## 2️⃣ 生成标注文件

运行自动标注脚本：

```bash
cd ~/Desktop/industrial_action_recognition
python src/prepare_data.py
```

这会自动：
- 扫描 `data/videos/` 下的所有视频
- 生成 `data/annotations/train_list.txt` (80% 数据)
- 生成 `data/annotations/val_list.txt` (20% 数据)

## 3️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

## 4️⃣ 训练模型（可选）

如果你有足够的训练数据：

```bash
python src/train.py --config configs/action_config.yaml
```

**注意：** 当前是框架版本，完整训练需要集成 PaddlePaddle PP-Human

## 5️⃣ 摄像头实时测试

```bash
python src/camera_demo.py
```

**快捷键：**
- `q` - 退出
- `r` - 重置帧缓冲区

## 6️⃣ 视频文件测试

```bash
python src/inference.py --video test.mp4 --output result.mp4
```

---

## 📝 修改动作类别

编辑 `configs/label_list.txt`，每行一个类别名：

```
welding
assembly
inspection
quality_check
packaging
```

然后重新运行 `prepare_data.py`

## 🎥 录制训练视频建议

1. **固定摄像头位置** - 保持视角一致
2. **充足光照** - 避免过暗或过曝
3. **完整动作** - 从开始到结束
4. **多样性** - 不同人、不同速度、不同角度
5. **背景简洁** - 减少干扰

## 🔧 调整配置

编辑 `configs/action_config.yaml`：

```yaml
num_classes: 5        # 改为你的类别数
epochs: 100           # 训练轮数
batch_size: 16        # 根据显存调整
learning_rate: 0.001  # 学习率
```

## ❓ 常见问题

**Q: 摄像头打不开？**
A: 检查摄像头权限，或尝试 `--camera 1` 切换摄像头

**Q: 识别不准确？**
A: 需要更多训练数据，建议每类 50+ 视频

**Q: 如何添加新动作？**
A: 在 `label_list.txt` 添加类别，在 `data/videos/` 创建对应文件夹

---

🎉 现在你可以开始收集训练数据了！
