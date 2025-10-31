# Image Filtering and Browsing Tools

这两个工具用于筛选和浏览数据集中的图片。

## 1. filter_new_2022s_images.py - 图片筛选工具

### 功能
- 从主数据集中筛选出 `new` 和 `2022s` 文件夹中的所有图片
- 复制图片到待处理文件夹，保持原有的艺术家/年份目录结构
- 自动复制对应的 JSON 文件（从 `jsons` 文件夹）
- 自动复制每个艺术家的 `results.json` 文件

### 使用方法

```bash
python tools/filter_new_2022s_images.py \
    --main-root /path/to/main/dataset \
    --output-root /path/to/filtered/output
```

### 参数说明
- `--main-root`: 主数据集的根目录（包含艺术家文件夹，每个艺术家文件夹内有年份子文件夹）
- `--output-root`: 输出目录，筛选后的图片将复制到这里

### 输出结构
```
output-root/
├── artist_1/
│   ├── results.json
│   ├── jsons/
│   │   ├── image1.json
│   │   └── image2.json
│   ├── new/
│   │   ├── image1.png
│   │   └── Augmentation/
│   │       └── aug_image1.png
│   └── 2022s/
│       └── image2.jpg
├── artist_2/
│   └── ...
```

### 示例
```bash
# 筛选图片
python tools/filter_new_2022s_images.py \
    --main-root "F:/AIGC/dataset/main" \
    --output-root "F:/AIGC/dataset/to_review"
```

---

## 2. image_browser.py - 交互式图片浏览器

### 功能
- 在浏览器中浏览指定目录下的所有图片
- 以艺术家为单位组织和显示图片
- 快捷键翻页（左右箭头）
- 点击图片或按钮快速删除不需要的图片
- 一键打开当前艺术家的文件夹

### 使用方法

```bash
python tools/image_browser.py \
    --root /path/to/filtered/output \
    --port 7860
```

### 参数说明
- `--root`: 要浏览的图片根目录（通常是第一步筛选后的输出目录）
- `--port`: Web服务器端口（默认：7860）
- `--share`: （可选）创建公共分享链接

### 快捷键
- `←` (左箭头): 上一张图片
- `→` (右箭头): 下一张图片
- `Delete`: 删除当前图片
- 点击图片: 删除当前图片

### 界面功能
- **Previous Artist / Next Artist**: 切换到上一个/下一个艺术家
- **Previous Image / Next Image**: 浏览当前艺术家的上一张/下一张图片
- **Delete Image**: 删除当前显示的图片（不可恢复！）
- **Open Artist Folder**: 在系统文件管理器中打开当前艺术家的文件夹

### 示例
```bash
# 启动图片浏览器
python tools/image_browser.py \
    --root "F:/AIGC/dataset/to_review" \
    --port 7860
```

浏览器会自动打开，访问地址：http://localhost:7860

---

## 完整工作流程示例

### 步骤 1: 筛选图片
```bash
python tools/filter_new_2022s_images.py \
    --main-root "F:/AIGC/train_lumina_konya_sd3_modified_dataset_format/dataset" \
    --output-root "F:/AIGC/dataset_review"
```

### 步骤 2: 浏览和筛选
```bash
python tools/image_browser.py \
    --root "F:/AIGC/dataset_review" \
    --port 7860
```

### 步骤 3: 处理筛选后的数据
筛选完成后，`F:/AIGC/dataset_review` 中保留的就是经过人工审核的高质量图片，可以继续用于训练或其他处理。

---

## 依赖库

确保已安装以下依赖：

```bash
pip install gradio pillow tqdm
```

---

## 注意事项

1. **备份重要数据**: 删除操作不可恢复，建议在筛选前备份原始数据
2. **磁盘空间**: 第一步会复制文件，确保有足够的磁盘空间
3. **浏览器兼容性**: 推荐使用 Chrome、Firefox 或 Edge 浏览器
4. **大数据集**: 对于包含大量图片的数据集，加载可能需要一些时间

---

## 故障排除

### 问题: 浏览器无法打开图片
- 检查图片文件是否损坏
- 确认图片格式是否支持（支持: png, jpg, jpeg, webp, bmp）

### 问题: 无法删除图片
- 检查文件权限
- 确认文件未被其他程序占用

### 问题: Web界面无法访问
- 检查端口是否被占用，尝试更换端口
- 检查防火墙设置

---

## 许可和使用

这些工具基于 `build_training_dataset.py` 开发，遵循相同的代码风格和约定。

