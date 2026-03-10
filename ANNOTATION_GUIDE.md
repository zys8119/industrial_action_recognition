# 图像标注工具使用指南

## 🎨 实时标注工具

用于在摄像头画面上实时标注物体位置和类别。

## 启动方式

```bash
cd ~/Desktop/industrial_action_recognition

# 方式1: 使用快捷脚本
./run_annotate.sh

# 方式2: 直接运行
source venv/bin/activate
python src/annotate_tool.py

# 指定摄像头
python src/annotate_tool.py --camera 1

# 指定输出目录
python src/annotate_tool.py --output my_annotations
```

## ⌨️ 操作说明

### 绘制标注框
1. **鼠标左键按住拖动** - 在物体周围绘制矩形框
2. 松开鼠标 - 完成标注

### 切换类别
- **数字键 1** - 切换到第1个类别（welding）
- **数字键 2** - 切换到第2个类别（assembly）
- **数字键 3** - 切换到第3个类别（inspection）

### 保存和管理
- **SPACE（空格）** - 保存当前帧和所有标注
- **C** - 清除当前帧的所有标注
- **U** - 撤销最后一个标注框
- **S** - 导出所有标注到 JSON 文件
- **H** - 显示帮助信息
- **Q** - 退出程序

## 📁 输出文件

标注数据保存在 `data/annotations/` 目录：

```
data/annotations/
├── frame_20260310_104800_123456.jpg    # 标注的图像
├── frame_20260310_104805_234567.jpg
└── annotations_20260310_104900.json    # 标注数据
```

## 📋 标注数据格式

JSON 文件包含所有标注信息：

```json
{
  "version": "1.0",
  "created_at": "20260310_104900",
  "labels": ["welding", "assembly", "inspection"],
  "total_frames": 10,
  "total_boxes": 25,
  "annotations": [
    {
      "image": "frame_20260310_104800_123456.jpg",
      "timestamp": "20260310_104800_123456",
      "width": 1280,
      "height": 720,
      "boxes": [
        {
          "x1": 100,
          "y1": 150,
          "x2": 300,
          "y2": 400,
          "label": "welding",
          "width": 200,
          "height": 250
        }
      ]
    }
  ]
}
```

## 💡 使用技巧

### 1. 标注工作流程
1. 启动工具，摄像头对准工作场景
2. 按数字键选择要标注的类别
3. 用鼠标框选物体位置
4. 按空格保存当前帧
5. 继续标注下一帧
6. 完成后按 S 导出，或退出时自动导出

### 2. 快速标注
- 先标注同一类别的所有物体，再切换类别
- 使用 U 键快速撤销错误标注
- 使用 C 键清除重新开始

### 3. 数据质量
- 确保边界框紧贴物体边缘
- 避免框选过大或过小
- 每个类别标注 50-100 帧
- 包含不同角度、距离、光照条件

### 4. 添加自定义类别
编辑 `configs/label_list.txt`：
```
welding
assembly
inspection
quality_check
packaging
```

## 🎯 标注示例场景

### 工业动作识别
- **焊接动作** - 框选焊枪和焊接区域
- **装配动作** - 框选手部和零件
- **检查动作** - 框选检测工具和被检物体

### 物体检测
- 框选工具、零件、产品
- 标注不同状态（正常/异常）

## 🔧 故障排除

**摄像头打不开？**
- 检查系统隐私设置中的摄像头权限
- 尝试 `--camera 1` 切换摄像头

**标注框太小无法保存？**
- 最小面积要求 100 像素
- 绘制更大的框

**颜色区分不清？**
- 每个类别有固定颜色
- 查看状态栏的当前类别提示

## 📊 后续使用

标注完成后，可以：
1. 用于训练目标检测模型（YOLO, Faster R-CNN）
2. 用于动作识别的关键区域定位
3. 分析工作流程和动作模式
