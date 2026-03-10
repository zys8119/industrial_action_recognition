# 项目工具总览

## 🛠️ 三大核心工具

### 1. 📹 动作识别 Demo (`camera_demo.py`)
**功能：** 实时识别工业动作，自动检测动作位置

**特点：**
- 自动检测动作发生的区域
- 实时显示边界框和置信度
- 记录详细日志
- 支持截图保存

**启动：**
```bash
./run_camera.sh
```

**适用场景：**
- 演示动作识别效果
- 测试模型性能
- 收集识别日志
- 实时监控

---

### 2. 🎨 静态标注工具 (`annotate_tool.py`)
**功能：** 手动标注图像，收集训练数据

**特点：**
- 鼠标拖动绘制边界框
- 多类别支持
- 自动保存标注数据
- 导出 JSON 格式

**启动：**
```bash
./run_annotate.sh
```

**适用场景：**
- 收集训练数据
- 标注数据集
- 质量检查
- 数据增强

---

### 3. 🎯 动态跟踪工具 (`tracking_tool.py`) ⭐ 新功能
**功能：** 手动标记后自动跟踪物体运动

**特点：**
- 标记后自动跟踪
- 显示运动轨迹
- 多目标同时跟踪
- 三种跟踪器可选
- 跟踪丢失自动检测

**启动：**
```bash
./run_tracking.sh
```

**适用场景：**
- 运动轨迹分析
- 工作流程研究
- 多目标监控
- 行为分析

---

## 📊 工具对比

| 功能 | 动作识别 | 静态标注 | 动态跟踪 |
|------|---------|---------|---------|
| 自动检测 | ✅ | ❌ | ❌ |
| 手动标记 | ❌ | ✅ | ✅ |
| 运动跟踪 | ❌ | ❌ | ✅ |
| 轨迹显示 | ❌ | ❌ | ✅ |
| 日志记录 | ✅ | ✅ | ✅ |
| 数据导出 | ❌ | ✅ | ✅ |
| 实时性 | 高 | 中 | 高 |

## 🎯 使用建议

### 场景 1：收集训练数据
1. 使用 **静态标注工具** 标注静态图像
2. 按 SPACE 保存每一帧
3. 导出 JSON 用于训练

### 场景 2：分析运动轨迹
1. 使用 **动态跟踪工具** 标记物体
2. 观察运动轨迹
3. 保存关键帧和轨迹数据

### 场景 3：实时监控识别
1. 使用 **动作识别 Demo** 实时检测
2. 查看日志分析结果
3. 截图保存异常情况

### 场景 4：混合使用
1. 先用 **动作识别** 找到感兴趣的区域
2. 用 **动态跟踪** 精确跟踪该区域
3. 用 **静态标注** 标注关键帧

## 🚀 快速开始

### 第一次使用
```bash
cd ~/Desktop/industrial_action_recognition

# 1. 激活虚拟环境
source venv/bin/activate

# 2. 选择一个工具启动
./run_tracking.sh      # 推荐：动态跟踪
./run_camera.sh        # 或：动作识别
./run_annotate.sh      # 或：静态标注
```

### 查看日志
```bash
# 动作识别日志
python src/view_log.py --analyze

# 跟踪日志
cat data/tracking/tracking_log.txt

# 实时监控
python src/view_log.py --watch
```

## 📁 输出目录

```
industrial_action_recognition/
├── logs/                    # 动作识别日志
│   └── recognition.log
├── screenshots/             # 截图
│   └── screenshot_*.jpg
├── data/
│   ├── annotations/        # 静态标注数据
│   │   ├── frame_*.jpg
│   │   └── annotations_*.json
│   └── tracking/           # 动态跟踪数据
│       ├── tracked_*.jpg
│       ├── tracked_*.json
│       └── tracking_log.txt
```

## 💡 专业提示

1. **性能优化**
   - 动作识别：使用 `--no-log` 禁用日志提升性能
   - 动态跟踪：使用 `--tracker MOSSE` 提升速度
   - 减少同时跟踪的目标数量

2. **数据质量**
   - 确保光照充足稳定
   - 避免剧烈运动和遮挡
   - 定期清理失效的跟踪目标

3. **工作流程**
   - 先用识别工具快速筛选
   - 再用跟踪工具精确分析
   - 最后用标注工具收集数据

4. **故障排查**
   - 摄像头权限问题 → 系统设置中授权
   - 性能卡顿 → 减少跟踪目标或切换跟踪器
   - 跟踪丢失 → 重新标记或改善环境

## 📚 详细文档

- [README.md](README.md) - 项目总览
- [QUICKSTART.md](QUICKSTART.md) - 快速开始
- [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) - 静态标注指南
- [TRACKING_GUIDE.md](TRACKING_GUIDE.md) - 动态跟踪指南
- [LOG_GUIDE.md](LOG_GUIDE.md) - 日志功能说明
