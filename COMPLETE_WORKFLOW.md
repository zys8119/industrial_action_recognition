# 完整工作流程：标注 → 训练 → 检测

## 🎯 三步完成物体识别系统

### 步骤 1️⃣：视频标注（采集数据）
```bash
cd ~/Desktop/industrial_action_recognition
./run_video_annotator.sh
```

**操作：**
1. 用鼠标框选要识别的物体
2. 按空格开始录制
3. 移动物体，系统自动跟踪标注
4. 按空格停止录制
5. 重复多次，收集不同角度和位置的数据

**建议：**
- 每个类别录制 2-3 个视频
- 每个视频 10-30 秒
- 包含不同角度、距离、光照

### 步骤 2️⃣：训练模型
```bash
source venv/bin/activate
python src/train_from_video.py
```

**效果：**
- 自动读取所有视频标注数据
- 提取物体特征
- 生成训练模型
- 保存到 `models/video_trained_model.pkl`

### 步骤 3️⃣：实时检测
```bash
python src/detect_from_video_model.py --model models/video_trained_model.pkl
```

**效果：**
- 加载训练好的模型
- 实时检测物体
- 显示位置、角度、尺寸

## 📊 完整示例

### 场景：识别螺栓和螺母

#### 1. 标注阶段
```bash
./run_video_annotator.sh
```

1. 框选螺栓（按1选择类别）
2. 按空格开始录制
3. 移动螺栓，自动跟踪标注
4. 按空格停止
5. 框选螺母（按2选择类别）
6. 重复录制过程

#### 2. 训练阶段
```bash
source venv/bin/activate
python src/train_from_video.py
```

输出示例：
```
🎓 开始训练...

📂 找到 2 个标注文件

处理: annotations_20260310_145900.json
✅ 提取了 45 个样本

处理: annotations_20260310_150200.json
✅ 提取了 38 个样本

📊 训练数据统计:
   螺栓: 45 个样本
   螺母: 38 个样本
   总计: 83 个样本

💾 模型已保存: models/video_trained_model.pkl
✅ 训练完成！
```

#### 3. 检测阶段
```bash
python src/detect_from_video_model.py --model models/video_trained_model.pkl
```

现在系统会自动识别螺栓和螺母！

## 🎨 可视化效果

### 检测显示
- **绿色框** - 高置信度 (>70%)
- **黄色框** - 中等置信度 (50-70%)
- **橙色框** - 低置信度 (<50%)

### 信息显示
```
螺栓 85%
Angle: 12.5°
Size: 200x195
```

## 📁 文件结构

```
data/video_annotations/
├── annotated_video_20260310_145900.mp4  # 标注视频
├── annotations_20260310_145900.json     # 标注数据
└── ...

models/
└── video_trained_model.pkl              # 训练好的模型

detection_20260310_150500.jpg            # 检测结果截图
```

## 💡 提高识别准确率

### 1. 收集更多数据
```bash
# 多录制几个视频
./run_video_annotator.sh  # 第1次
./run_video_annotator.sh  # 第2次
./run_video_annotator.sh  # 第3次

# 然后重新训练
python src/train_from_video.py
```

### 2. 包含多样性
- ✅ 不同角度（正面、侧面、斜面）
- ✅ 不同距离（近、中、远）
- ✅ 不同光照（明亮、正常、较暗）
- ✅ 不同背景

### 3. 标注质量
- ✅ 框选准确，紧贴物体
- ✅ 物体清晰可见
- ✅ 避免严重遮挡
- ✅ 移动平滑，不要太快

## 🔧 调整参数

### 降低误检（提高精度）
```bash
python src/detect_from_video_model.py \
    --model models/video_trained_model.pkl \
    --min-matches 15 \
    --confidence 0.7
```

### 提高检出率（降低漏检）
```bash
python src/detect_from_video_model.py \
    --model models/video_trained_model.pkl \
    --min-matches 5 \
    --confidence 0.4
```

## 📊 数据统计

### 查看训练数据
```python
import pickle

with open('models/video_trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"类别: {model['labels']}")
print(f"总样本: {model['total_samples']}")

for label, templates in model['templates'].items():
    print(f"{label}: {len(templates)} 个样本")
```

### 分析标注数据
```python
import json

with open('data/video_annotations/annotations_20260310_145900.json') as f:
    data = json.load(f)

print(f"总帧数: {data['total_frames']}")
print(f"类别: {data['labels']}")

# 统计每个类别的标注数量
label_counts = {}
for frame in data['frames']:
    for ann in frame['annotations']:
        label = ann['label']
        label_counts[label] = label_counts.get(label, 0) + 1

for label, count in label_counts.items():
    print(f"{label}: {count} 次标注")
```

## 🆚 与其他方法对比

| 方法 | 数据需求 | 训练时间 | 准确率 | 实时性 |
|------|---------|---------|--------|--------|
| 深度学习 | 数百张 | 数小时 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 视频标注训练 | 2-3个视频 | 无需训练 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 单图标注 | 5-10张 | 无需训练 | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🎯 适用场景

### ✅ 适合
- 固定形状的物体
- 纹理丰富的物体
- 小批量多品种
- 快速部署
- 数据收集困难

### ❌ 不适合
- 形变大的物体
- 纹理单一的物体
- 需要高精度（>95%）
- 大规模生产线

## 🚀 快速命令

```bash
# 完整流程
./run_video_annotator.sh              # 1. 标注
source venv/bin/activate
python src/train_from_video.py        # 2. 训练
python src/detect_from_video_model.py \
    --model models/video_trained_model.pkl  # 3. 检测
```

## 📈 持续改进

### 添加新类别
1. 运行标注工具，框选新类别
2. 重新训练：`python src/train_from_video.py`
3. 新模型会包含所有类别

### 提高现有类别准确率
1. 录制更多该类别的视频
2. 重新训练
3. 样本越多，准确率越高

### 更新模型
```bash
# 删除旧模型
rm models/video_trained_model.pkl

# 重新训练
python src/train_from_video.py
```

现在你有了完整的工作流程：标注 → 训练 → 检测！
