# 工业物体动作识别 Demo

基于 PaddlePaddle PP-Human 的工业场景动作识别系统

## 项目结构

```
industrial_action_recognition/
├── data/                    # 训练数据目录
│   ├── videos/             # 原始视频文件
│   ├── annotations/        # 标注文件
│   └── processed/          # 处理后的数据
├── models/                 # 模型文件
├── configs/                # 配置文件
├── src/                    # 源代码
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   └── camera_demo.py     # 摄像头实时识别
├── requirements.txt        # 依赖包
└── README.md              # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 准备训练数据

### 1. 视频数据格式

将训练视频放入 `data/videos/` 目录，按动作类别组织：

```
data/videos/
├── action1_welding/        # 焊接动作
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── action2_assembly/       # 装配动作
│   ├── video1.mp4
│   └── ...
└── action3_inspection/     # 检查动作
    └── ...
```

### 2. 标注格式

在 `data/annotations/` 目录创建标注文件 `train_list.txt`：

```
videos/action1_welding/video1.mp4 0
videos/action1_welding/video2.mp4 0
videos/action2_assembly/video1.mp4 1
videos/action3_inspection/video1.mp4 2
```

格式：`视频路径 类别ID`

### 3. 类别配置

编辑 `configs/label_list.txt`，每行一个动作类别：

```
welding
assembly
inspection
```

## 使用方法

### 训练模型

```bash
python src/train.py --config configs/action_config.yaml
```

### 摄像头实时识别

```bash
python src/camera_demo.py --model models/best_model.pdparams
```

### 视频文件识别

```bash
python src/inference.py --video test.mp4 --model models/best_model.pdparams
```

## 快速开始

1. 准备数据：将视频按类别放入 `data/videos/` 对应文件夹
2. 创建标注：运行 `python src/prepare_data.py` 自动生成标注文件
3. 训练模型：`python src/train.py`
4. 实时测试：`python src/camera_demo.py`

## 注意事项

- 每个动作类别建议至少准备 20-50 个视频样本
- 视频时长建议 3-10 秒
- 确保摄像头权限已开启
- 首次运行会自动下载预训练模型
