# 训练数据准备说明

## ❓ 为什么提示"没有训练样本"？

### 常见原因

1. **视频文件位置不对**
   - ❌ 错误：`data/videos/video1.mp4`
   - ✅ 正确：`data/videos/搬运箱子/video1.mp4`
   
   视频必须放在**类别文件夹**下！

2. **类别文件夹名称不匹配**
   - `configs/label_list.txt` 中的类别名
   - 必须与 `data/videos/` 下的文件夹名**完全一致**
   
   例如：
   ```
   label_list.txt 内容：搬运箱子
   文件夹名称：data/videos/搬运箱子/
   ```

3. **视频格式不支持**
   - 支持的格式：`.mp4`, `.avi`, `.mov`（大小写均可）
   - 不支持的格式：`.mkv`, `.flv`, `.wmv` 等

4. **样本数量太少**
   - 只有 1 个视频时，会全部用于训练（验证集为 0）
   - 少于 5 个视频时，全部用于训练
   - 5 个或以上时，按 80/20 分割

## ✅ 正确的目录结构

```
industrial_action_recognition/
├── configs/
│   └── label_list.txt          # 类别列表
│       内容：
│       搬运箱子
│       焊接
│       装配
│
└── data/
    └── videos/
        ├── 搬运箱子/            # 类别1
        │   ├── video1.mp4
        │   ├── video2.mp4
        │   └── video3.mp4
        ├── 焊接/                # 类别2
        │   ├── video1.mp4
        │   └── video2.mp4
        └── 装配/                # 类别3
            └── video1.mp4
```

## 🔧 解决步骤

### 1. 检查类别配置

```bash
cd ~/Desktop/industrial_action_recognition
cat configs/label_list.txt
```

输出应该是你的动作类别，每行一个：
```
搬运箱子
焊接
装配
```

### 2. 检查视频文件

```bash
# 查看所有视频文件
find data/videos -type f -name "*.mp4"
```

输出示例：
```
data/videos/搬运箱子/video1.mp4
data/videos/搬运箱子/video2.mp4
data/videos/焊接/video1.mp4
```

### 3. 创建类别文件夹

如果文件夹不存在，创建它们：

```bash
# 根据 label_list.txt 创建文件夹
cd data/videos
mkdir -p 搬运箱子 焊接 装配
```

### 4. 移动视频文件

将视频文件移动到对应的类别文件夹：

```bash
# 例如：将视频移动到"搬运箱子"文件夹
mv video1.mp4 搬运箱子/
mv video2.mp4 搬运箱子/
```

### 5. 运行数据准备脚本

```bash
cd ~/Desktop/industrial_action_recognition
source venv/bin/activate
python src/prepare_data.py
```

成功输出示例：
```
📋 发现 3 个动作类别: ['搬运箱子', '焊接', '装配']
📹 搬运箱子: 找到 10 个视频
📹 焊接: 找到 8 个视频
📹 装配: 找到 5 个视频
   ⚠️  样本较少，全部用于训练

✅ 数据准备完成！
   训练样本: 18
   验证样本: 5
```

### 6. 检查生成的标注文件

```bash
cat data/annotations/train_list.txt
```

输出示例：
```
videos/搬运箱子/video1.mp4 0
videos/搬运箱子/video2.mp4 0
videos/焊接/video1.mp4 1
videos/装配/video1.mp4 2
```

格式：`视频路径 类别ID`

## 📊 样本数量建议

| 样本数量 | 状态 | 说明 |
|---------|------|------|
| 1-4 个 | ⚠️ 太少 | 全部用于训练，无验证集 |
| 5-19 个 | ⚠️ 较少 | 可以训练，但效果可能不佳 |
| 20-49 个 | ✅ 基本 | 可以训练，效果一般 |
| 50+ 个 | ✅ 推荐 | 训练效果较好 |
| 100+ 个 | 🎯 理想 | 训练效果很好 |

**建议：每个类别至少准备 20-50 个视频**

## 🎥 视频要求

### 时长
- 推荐：3-10 秒
- 最短：1 秒
- 最长：30 秒

### 内容
- 清晰展示完整动作
- 避免过度遮挡
- 光照充足
- 背景相对简洁

### 多样性
- 不同角度
- 不同距离
- 不同光照条件
- 不同执行者

## 🔍 常见错误示例

### 错误 1：视频直接放在 videos 目录
```
❌ data/videos/video1.mp4
✅ data/videos/搬运箱子/video1.mp4
```

### 错误 2：文件夹名称不匹配
```
label_list.txt: 搬运箱子
文件夹名称: 搬运箱子1  ❌ 不匹配！
```

### 错误 3：使用英文类别但文件夹是中文
```
label_list.txt: carry_box
文件夹名称: 搬运箱子  ❌ 不匹配！
```

### 错误 4：视频格式不支持
```
❌ video.mkv
❌ video.flv
✅ video.mp4
✅ video.avi
✅ video.mov
```

## 💡 快速测试

创建测试数据：

```bash
cd ~/Desktop/industrial_action_recognition

# 1. 编辑类别
echo "test_action" > configs/label_list.txt

# 2. 创建文件夹
mkdir -p data/videos/test_action

# 3. 复制一个测试视频（如果有）
# cp ~/Downloads/test.mp4 data/videos/test_action/

# 4. 运行准备脚本
source venv/bin/activate
python src/prepare_data.py
```

## 📞 仍然有问题？

运行诊断命令：

```bash
cd ~/Desktop/industrial_action_recognition

echo "=== 类别配置 ==="
cat configs/label_list.txt

echo -e "\n=== 视频文件 ==="
find data/videos -type f -name "*.mp4" -o -name "*.avi" -o -name "*.mov"

echo -e "\n=== 文件夹结构 ==="
ls -R data/videos/

echo -e "\n=== 标注文件 ==="
cat data/annotations/train_list.txt 2>/dev/null || echo "标注文件不存在"
```

将输出结果发给我，我可以帮你诊断问题。
