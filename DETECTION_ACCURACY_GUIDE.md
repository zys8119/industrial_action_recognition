# 检测不准确问题排查

## 🔍 问题现象

检测器能工作，但是识别到的不是标注的物体。

## 📊 当前状态

- ✅ 训练数据：494 个标注
- ✅ 类别：yazi
- ✅ 检测器：工作正常
- ❌ 问题：识别错误的物体

## 🎯 排查步骤

### 步骤 1：查看训练样本

```bash
cd ~/Desktop/industrial_action_recognition

# 查看提取的训练样本
open training_samples_preview/
```

**检查内容：**
1. 打开 `frame_000.jpg` 到 `frame_019.jpg`
2. 查看绿色框是否准确框住了你想要的物体
3. 查看 `sample_XXX_yazi.jpg` 是否是正确的物体图像

### 步骤 2：确认问题

**如果标注正确：**
- 绿色框准确框住了目标物体
- 提取的图像是正确的物体
- → 继续步骤 3

**如果标注错误：**
- 绿色框框住了错误的物体
- 或者框选不准确
- → 需要重新录制视频（跳到步骤 4）

### 步骤 3：测试检测

将标注的物体放在摄像头前：

```bash
./run_simple_detector.sh
```

**测试方法：**
1. 将训练时标注的物体放在摄像头前
2. 保持与训练时相似的角度和距离
3. 观察是否能检测到

**如果能检测到：**
- ✅ 系统工作正常
- 只是当前画面中没有训练的物体

**如果检测到其他物体：**
- 可能是误检
- 或者训练数据中混入了其他物体

### 步骤 4：重新录制（如果需要）

如果标注有问题，重新录制：

```bash
# 1. 清除旧数据（可选）
rm -rf data/video_annotations/annotated_video_*.mp4
rm -rf data/video_annotations/annotations_*.json

# 2. 重新标注
./run_video_annotator.sh

# 操作要点：
# - 仔细框选目标物体
# - 确保框选准确
# - 避免框选背景
# - 物体要清晰可见

# 3. 重新训练
source venv/bin/activate
python src/train_from_video.py

# 4. 重新检测
./run_simple_detector.sh
```

## 💡 常见问题

### 问题 1：标注时框选不准确

**原因：**
- 框太大，包含了背景
- 框太小，只包含部分物体
- 跟踪丢失，框选了错误位置

**解决：**
- 第一次框选时要准确
- 物体移动不要太快
- 保持物体在画面中央
- 避免遮挡

### 问题 2：训练了多个物体

**原因：**
- 录制时画面中有多个物体
- 跟踪器跟踪了错误的物体

**解决：**
- 录制时只放一个物体
- 背景要简洁
- 删除错误的跟踪（按 D 键）

### 问题 3：检测到相似的物体

**原因：**
- 特征相似的物体会被误识别
- 训练样本不够多样化

**解决：**
- 增加训练样本
- 包含更多角度和位置
- 使用特征更明显的物体

## 🎯 最佳实践

### 录制视频时

1. **环境准备**
   - 简洁的背景
   - 充足的光照
   - 固定摄像头

2. **物体准备**
   - 只放一个物体
   - 物体清晰可见
   - 特征明显

3. **录制过程**
   - 第一次框选要准确
   - 移动要平滑
   - 包含多个角度
   - 录制 20-30 秒

4. **质量检查**
   - 按 S 键保存几帧
   - 检查标注是否准确
   - 如果不准确，按 C 清除重来

### 训练后

1. **验证训练数据**
   ```bash
   python src/visualize_training.py
   open training_samples_preview/
   ```

2. **测试检测**
   - 使用训练时的物体测试
   - 保持相似的环境
   - 观察检测效果

3. **迭代改进**
   - 如果效果不好，录制更多视频
   - 重新训练
   - 再次测试

## 📝 检查清单

在重新录制前，确认：

- [ ] 我知道要标注什么物体
- [ ] 物体有明显的纹理特征
- [ ] 背景简洁
- [ ] 光照充足
- [ ] 摄像头固定
- [ ] 只有一个物体在画面中

录制时：

- [ ] 第一次框选准确
- [ ] 物体完全在框内
- [ ] 移动平滑
- [ ] 包含多个角度
- [ ] 录制 20-30 秒

录制后：

- [ ] 运行 `python src/visualize_training.py`
- [ ] 检查 `training_samples_preview/` 中的图像
- [ ] 确认标注正确
- [ ] 重新训练
- [ ] 测试检测

## 🚀 快速修复

如果你确认标注有问题，最快的方法：

```bash
cd ~/Desktop/industrial_action_recognition

# 1. 删除旧数据
rm -rf data/video_annotations/*.mp4
rm -rf data/video_annotations/*.json
rm -rf models/video_trained_model.pkl

# 2. 重新开始
./run_video_annotator.sh
# 仔细标注，录制 20-30 秒

# 3. 训练
source venv/bin/activate
python src/train_from_video.py

# 4. 检查训练样本
python src/visualize_training.py
open training_samples_preview/

# 5. 如果样本正确，开始检测
./run_simple_detector.sh
```

现在请检查 `training_samples_preview/` 目录中的图像，确认标注是否正确！
