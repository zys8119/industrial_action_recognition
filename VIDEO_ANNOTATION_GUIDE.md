# 视频自动标注工具使用指南

## 🎬 功能介绍

这是一个**智能视频标注工具**，只需第一次手动框选物体，后续会自动跟踪并连续标注。

### 核心特性
- ✅ **一次框选，自动跟踪** - 手动标注一次，自动跟踪整个视频
- ✅ **多目标跟踪** - 同时跟踪多个物体
- ✅ **轨迹显示** - 显示物体运动轨迹
- ✅ **视频录制** - 录制带标注的视频
- ✅ **数据导出** - 导出每帧的标注数据（JSON）

## 🚀 使用流程

### 步骤 1️⃣：启动工具
```bash
cd ~/Desktop/industrial_action_recognition
./run_video_annotator.sh
```

### 步骤 2️⃣：框选物体
1. 用鼠标拖动框选要跟踪的物体
2. 松开鼠标，系统自动开始跟踪
3. 可以框选多个物体

### 步骤 3️⃣：开始录制
1. 按**空格键**开始录制
2. 屏幕显示 🔴 REC 表示正在录制
3. 物体移动时自动标注

### 步骤 4️⃣：停止录制
1. 再按**空格键**停止录制
2. 自动保存视频和标注数据

## ⌨️ 快捷键

### 基本操作
- **鼠标拖动** - 框选物体（自动开始跟踪）
- **数字键 1-9** - 切换类别
- **SPACE（空格）** - 开始/停止录制

### 显示控制
- **T** - 切换轨迹显示（开/关）
- **I** - 切换ID显示（开/关）

### 管理操作
- **C** - 清除所有跟踪
- **D** - 删除最后一个跟踪
- **S** - 保存当前帧（不录制）
- **H** - 显示帮助
- **Q** - 退出

## 📁 输出文件

标注数据保存在 `data/video_annotations/` 目录：

```
data/video_annotations/
├── annotated_video_20260310_144300.mp4  # 带标注的视频
├── annotations_20260310_144300.json     # 标注数据
├── frame_20260310_144500.jpg            # 单帧截图
├── frame_20260310_144500.json           # 单帧标注
└── auto_annotation_log.txt              # 操作日志
```

## 📊 标注数据格式

### 视频标注数据（JSON）
```json
{
  "timestamp": "20260310_144300",
  "total_frames": 300,
  "labels": ["螺栓", "螺母", "垫片"],
  "frames": [
    {
      "frame": 1,
      "annotations": [
        {
          "id": 1,
          "label": "螺栓",
          "bbox": [120, 180, 200, 200],
          "center": [220, 280]
        },
        {
          "id": 2,
          "label": "螺母",
          "bbox": [450, 200, 150, 150],
          "center": [525, 275]
        }
      ]
    },
    {
      "frame": 2,
      "annotations": [
        {
          "id": 1,
          "label": "螺栓",
          "bbox": [125, 185, 200, 200],
          "center": [225, 285]
        }
      ]
    }
  ]
}
```

### 数据说明
- **id** - 物体唯一标识（跟踪ID）
- **label** - 物体类别
- **bbox** - 边界框 [x, y, width, height]
- **center** - 中心点坐标 [x, y]

## 💡 使用场景

### 1. 工业动作分析
- 跟踪工人的手部动作
- 记录工具的使用轨迹
- 分析装配流程

### 2. 物体运动分析
- 记录物体移动路径
- 分析运动速度
- 检测异常运动

### 3. 训练数据收集
- 快速收集大量标注数据
- 用于训练目标检测模型
- 用于训练跟踪算法

### 4. 质量检测
- 跟踪产品在流水线上的移动
- 记录检测过程
- 生成检测报告

## 🎯 实际应用示例

### 示例 1：装配线跟踪

```bash
# 1. 启动工具
./run_video_annotator.sh

# 2. 框选传送带上的产品
#    - 用鼠标框选第一个产品
#    - 按数字键选择类别

# 3. 开始录制
#    - 按空格键开始
#    - 产品移动时自动跟踪标注

# 4. 停止录制
#    - 按空格键停止
#    - 获得完整的运动轨迹数据
```

### 示例 2：多物体跟踪

```bash
# 1. 框选多个物体
#    - 框选螺栓（按1选择类别）
#    - 框选螺母（按2选择类别）
#    - 框选垫片（按3选择类别）

# 2. 开始录制
#    - 所有物体同时跟踪
#    - 自动标注每个物体的位置
```

## 📈 数据分析

### 分析运动轨迹

```python
import json

# 读取标注数据
with open('annotations_20260310_144300.json') as f:
    data = json.load(f)

# 提取某个物体的轨迹
obj_id = 1
trajectory = []

for frame_data in data['frames']:
    for ann in frame_data['annotations']:
        if ann['id'] == obj_id:
            trajectory.append(ann['center'])

print(f"物体 #{obj_id} 的轨迹: {len(trajectory)} 个点")

# 计算移动距离
import numpy as np
points = np.array(trajectory)
distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
total_distance = np.sum(distances)
print(f"总移动距离: {total_distance:.0f} 像素")
```

### 统计物体出现次数

```python
# 统计每个物体出现的帧数
object_frames = {}

for frame_data in data['frames']:
    for ann in frame_data['annotations']:
        obj_id = ann['id']
        if obj_id not in object_frames:
            object_frames[obj_id] = 0
        object_frames[obj_id] += 1

for obj_id, count in object_frames.items():
    print(f"物体 #{obj_id}: 出现 {count} 帧")
```

## 🔧 高级技巧

### 1. 提高跟踪准确性
- 确保物体特征清晰
- 避免快速移动
- 保持光照稳定
- 减少遮挡

### 2. 处理跟踪丢失
- 如果跟踪丢失，按 D 删除
- 重新框选物体
- 继续录制

### 3. 批量处理
```bash
# 录制多个视频
for i in {1..5}; do
    echo "录制视频 $i"
    ./run_video_annotator.sh
done
```

### 4. 转换为训练数据
```python
# 将标注数据转换为 YOLO 格式
def convert_to_yolo(json_file, output_dir):
    with open(json_file) as f:
        data = json.load(f)
    
    for frame_data in data['frames']:
        frame_num = frame_data['frame']
        
        # 生成 YOLO 标注文件
        yolo_lines = []
        for ann in frame_data['annotations']:
            # 转换为 YOLO 格式
            # class_id x_center y_center width height (归一化)
            pass
```

## 🎨 可视化

### 显示元素
- **彩色边界框** - 不同类别用不同颜色
- **物体ID** - 显示跟踪编号
- **运动轨迹** - 彩色线条显示路径
- **中心点** - 标记物体中心

### 状态栏信息
- 当前选择的类别
- 正在跟踪的物体数量
- 录制状态（🔴 REC）
- 当前帧数

## ⚠️ 注意事项

### 跟踪限制
- 物体不能完全离开画面
- 避免严重遮挡
- 不适合快速运动（>30像素/帧）
- 背景不要太复杂

### 性能优化
- 同时跟踪不超过 5-10 个物体
- 降低视频分辨率可提高速度
- 使用 SSD 存储提高写入速度

### 数据管理
- 定期清理旧视频
- 备份重要标注数据
- 使用有意义的文件名

## 🆚 与其他工具对比

| 工具 | 标注方式 | 适用场景 |
|------|---------|---------|
| annotate_tool.py | 逐帧手动 | 静态图像标注 |
| tracking_tool.py | 手动+跟踪 | 实时跟踪演示 |
| video_auto_annotator.py | 自动跟踪+录制 | 视频数据收集 ✅ |

## 📚 后续使用

标注完成后，数据可用于：
1. 训练目标检测模型（YOLO, Faster R-CNN）
2. 训练目标跟踪模型（DeepSORT, FairMOT）
3. 运动分析和统计
4. 质量检测和异常检测
5. 生成训练视频

现在你可以快速收集大量带标注的视频数据了！
