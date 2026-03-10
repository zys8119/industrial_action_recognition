# 视觉检查工具使用指南

## 🔍 功能说明

这是一个基于参考图片的视觉检查工具。你只需要提供一张目标物体的参考图片，工具会自动学习其特征，然后在摄像头或其他图片中检测相同的物体。

### 核心特性
- ✅ **零训练** - 无需训练模型，提供参考图片即可
- ✅ **特征匹配** - 基于 SIFT/ORB 特征点匹配
- ✅ **实时检测** - 摄像头实时检测
- ✅ **精确定位** - 显示物体位置和轮廓
- ✅ **置信度评估** - 显示匹配质量

## 🚀 快速开始

### 1. 准备参考图片

拍摄或准备一张清晰的目标物体图片，例如：
- 螺栓
- 零件
- 产品
- 缺陷样本

保存为 `reference.jpg`

### 2. 运行检测

#### 摄像头实时检测
```bash
cd ~/Desktop/industrial_action_recognition

# 基本用法
./run_inspector.sh --template reference.jpg

# 指定模板名称
./run_inspector.sh --template bolt.jpg --name "螺栓"

# 指定摄像头
./run_inspector.sh --template reference.jpg --camera 1
```

#### 检查单张图片
```bash
# 检查图片中是否有目标物体
./run_inspector.sh --template reference.jpg --image test.jpg
```

## 📋 使用示例

### 示例 1：检测螺栓

```bash
# 1. 拍摄一张螺栓的清晰照片作为参考
# 保存为 bolt_reference.jpg

# 2. 运行检测
./run_inspector.sh --template bolt_reference.jpg --name "M8螺栓"

# 3. 将摄像头对准工作台，系统会自动检测所有相似的螺栓
```

### 示例 2：质量检查

```bash
# 1. 准备合格产品的参考图片
./run_inspector.sh --template good_product.jpg --name "合格品"

# 2. 检查待检产品
./run_inspector.sh --template good_product.jpg --image product_001.jpg
```

### 示例 3：缺陷检测

```bash
# 1. 拍摄缺陷样本作为参考
./run_inspector.sh --template defect_sample.jpg --name "裂纹"

# 2. 实时检测是否有相似缺陷
```

## 🎨 界面说明

### 检测框颜色
- **绿色** - 高置信度 (>70%)
- **黄色** - 中等置信度 (50-70%)
- **橙色** - 低置信度 (<50%)

### 显示信息
- **边界框** - 矩形框标注检测位置
- **多边形轮廓** - 精确的物体轮廓
- **标签** - 模板名称 + 置信度
- **匹配点数** - 特征匹配数量

### 状态栏
- Templates: 加载的模板数量
- Detections: 当前检测到的物体数量

## ⌨️ 快捷键

- **Q** - 退出程序
- **S** - 保存当前帧（包含检测结果）

## 📁 输出文件

检测结果保存在 `data/inspection/` 目录：

```
data/inspection/
├── template_螺栓_features.jpg      # 模板特征可视化
├── detection_20260310_132400.jpg   # 检测结果截图
└── inspection_log.txt              # 检测日志
```

## 💡 最佳实践

### 拍摄参考图片
1. **清晰对焦** - 确保物体清晰
2. **充足光照** - 避免阴影和反光
3. **合适角度** - 与实际检测角度相似
4. **填充画面** - 物体占据画面主要部分
5. **简洁背景** - 减少背景干扰

### 提高检测准确率
1. **多角度模板** - 为同一物体准备多个角度的参考图
2. **相似光照** - 保持参考图和检测环境光照一致
3. **固定距离** - 保持摄像头到物体的距离相对固定
4. **清洁镜头** - 确保摄像头镜头清洁

### 性能优化
- 工具每 5 帧检测一次，平衡性能和实时性
- 如果检测太慢，可以降低图片分辨率
- 如果误检太多，提高 `min_matches` 参数

## 🔧 高级用法

### 多模板检测

创建一个脚本同时加载多个模板：

```python
from src.visual_inspector import VisualInspector

inspector = VisualInspector()
inspector.add_template("bolt.jpg", "螺栓")
inspector.add_template("nut.jpg", "螺母")
inspector.add_template("washer.jpg", "垫片")
inspector.run_camera()
```

### 调整检测参数

```python
# 更严格的匹配（减少误检）
detections = inspector.detect_in_frame(frame, 
    min_matches=20,      # 最少匹配点数
    match_threshold=0.6  # 匹配阈值
)

# 更宽松的匹配（增加检出率）
detections = inspector.detect_in_frame(frame,
    min_matches=5,
    match_threshold=0.8
)
```

## 📊 工作原理

1. **特征提取** - 使用 SIFT 算法提取参考图片的特征点
2. **特征匹配** - 在待检图片中寻找相似的特征点
3. **几何验证** - 使用 RANSAC 算法验证匹配的几何一致性
4. **位置计算** - 计算物体的精确位置和轮廓

## 🆚 与深度学习方法对比

| 特性 | 视觉检查工具 | 深度学习 |
|------|------------|---------|
| 训练时间 | 无需训练 | 需要数小时 |
| 数据需求 | 1张参考图 | 数百张标注图 |
| 准确率 | 中等 | 高 |
| 适用场景 | 固定物体、清晰特征 | 复杂场景、变化大 |
| 计算资源 | 低 | 高（需要GPU） |

## 🎯 适用场景

### ✅ 适合
- 零件识别和定位
- 产品一致性检查
- 简单缺陷检测
- 快速原型验证
- 小批量检测

### ❌ 不适合
- 高度变形的物体
- 光照变化剧烈
- 遮挡严重
- 需要语义理解
- 大规模生产线（建议用深度学习）

## 🔍 故障排除

**检测不到物体？**
- 检查参考图片是否清晰
- 确保光照条件相似
- 尝试更近的距离
- 检查物体是否有足够的纹理特征

**误检太多？**
- 提高 `min_matches` 参数
- 降低 `match_threshold`
- 使用更清晰的参考图片
- 简化背景

**检测太慢？**
- 降低图片分辨率
- 增加检测间隔（修改代码中的 `frame_count % 5`）
- 使用 ORB 代替 SIFT（更快但不太准确）

## 📈 后续改进

可以扩展的功能：
- 支持多模板批量检测
- 添加缺陷分类
- 集成测量功能（尺寸、角度）
- 导出检测报告
- 与数据库集成
