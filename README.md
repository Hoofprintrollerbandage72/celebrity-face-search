# 公众人物人脸检索 MVP

一个收敛到核心流程的本地系统：

1. 建立人物及参考图片库；
2. 上传待检索图片；
3. 检测图片中的人脸并提取向量；
4. 返回 Top-K 相似人物和参考图。

系统不采集使用场景，也不输出法律结论。相似度只表示模型特征接近程度。

## 已实现

- 浏览器上传、拖放和图片预览；
- 多人脸查询结果结构；
- 人物创建、删除和参考图上传；
- 图片源引导与一键下载、单人脸校验、建档和关联；
- SQLite 元数据和本地图片存储；
- 内存 NumPy 余弦相似度索引；
- 按人物聚合多张参考图片；
- DeepFace 可插拔识别引擎；
- 批量目录导入脚本；
- CelebA / VGGFace2 可断点批量导入、数据集计数和索引重载；
- 文件类型、大小及空库检查；
- API 自动文档和基础测试。

## 一键启动真实 MVP

```bash
./scripts/start_mvp.sh
```

脚本会检查 Python 依赖、下载并校验 OpenCV YuNet + SFace 模型，然后在
<http://127.0.0.1:8000> 启动服务。首次执行需要下载约 37 MB 模型文件。

模型来源于 OpenCV 官方模型库。YuNet 目录为 MIT，SFace 目录为 Apache 2.0。

启动后可在另一个终端执行真实闭环演示：

```bash
source .venv/bin/activate
python scripts/seed_demo_library.py
```

脚本会从 Wikimedia Commons 下载两位公众人物的带来源参考图，录入人物库，
再使用不同年份的 Donald Trump 图片检索并验证 Top-1。演示文件及模型都位于
被 Git 忽略的 `data/` 目录。

## 其他运行模式

### 1. 立即运行界面和流程

`demo` 引擎使用整图颜色与灰度特征，仅用于测试系统流程，不具备真实身份识别能力。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
FACE_ENGINE=demo uvicorn app.main:app --reload
```

访问 <http://localhost:8000>，API 文档位于 <http://localhost:8000/docs>。

### 2. DeepFace 可选引擎

如需对比 DeepFace，可通过 Docker 使用固定的 Python 3.11 环境：

```bash
docker compose up --build
```

第一次提取人脸时 DeepFace 可能下载所选模型权重。下载完成后即可录入参考图并进行检索。

也可以在兼容的 Python 环境中运行：

```bash
pip install -r requirements-deepface.txt
FACE_ENGINE=deepface uvicorn app.main:app --reload
```

## 建立人物库

可以通过网页“人物库”页面逐个录入，也可以准备以下目录：

### 图片源一键配置

打开“人物库 → 图片源快速配置”，选择一个已有人物，或填写姓名以同时创建
新人物。图片源每行填写一个，竖线右侧可选填可核验的来源页面：

```text
https://images.example.org/person/portrait-01.jpg | https://example.org/photo-page-01
https://images.example.org/person/portrait-02.jpg | https://example.org/photo-page-02
```

填写许可证或授权备注后点击“一键下载并关联”。系统会逐张完成：

1. 校验公开 HTTP(S) 地址并拦截本机、内网和保留地址；
2. 限制下载大小和跳转次数，验证 JPEG、PNG 或 WEBP 文件；
3. 要求参考图恰好检测到一张人脸；
4. 保存原图到本地、写入来源和授权信息、生成向量并关联人物；
5. 一次重载索引，并汇总成功、重复跳过和失败条目。

同一人物重复提交同一图片直链时会自动跳过，不会重复入库。每次最多处理
20 个图片源。接口说明可查看 `GET /api/library/source-guide` 和
`POST /api/library/quick-source-import`。

> 图片直链是服务器实际下载的地址；来源页面用于人工回溯作者、许可证和上下文。
> 记录来源不代表自动获得肖像权、版权或生物识别数据处理授权。

### 本地目录批量导入

```text
my-library/
├── Person A/
│   ├── 01.jpg
│   └── 02.jpg
└── Person B/
    ├── 01.jpg
    └── 02.jpg
```

然后批量导入：

```bash
python3 scripts/import_library.py ./my-library --license "CC BY 4.0"
```

参考图在真实人脸引擎下必须恰好检测到一张人脸。每位人物建议录入 3—10 张不同年龄、角度和妆容的照片。

### 导入 CelebA 与 VGGFace2

先录入两个数据集的全部身份元数据：

```bash
source .venv/bin/activate
python scripts/import_research_datasets.py bootstrap --dataset all
```

默认每个身份选取 5 张能成功检测到人脸的参考图。CelebA 从 Hugging Face
Parquet 流式读取并保存断点：

```bash
python scripts/import_research_datasets.py celeba --images-per-identity 5
```

VGGFace2 先断点下载约 40.2 GB 的两个归档，再流式遍历归档，不展开全部
330 万张图片，只保留每人需要的参考图：

```bash
python scripts/import_research_datasets.py download-vgg --which all
python scripts/import_research_datasets.py vggface2 --which all --images-per-identity 5
```

查看实际写入数量：

```bash
python scripts/import_research_datasets.py status
curl http://127.0.0.1:8000/api/library/stats
```

CelebA 只公开匿名 `celeb_id`，因此库内名称为 `CelebA #00001`；VGGFace2
使用 `identity_meta.csv` 中的姓名。详细来源、限制、断点文件和恢复方法见
[`docs/数据集导入说明.md`](docs/数据集导入说明.md)。

## 配置

复制 `.env.example` 或通过环境变量配置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `APP_DATA_DIR` | `./data` | SQLite 和参考图片目录 |
| `FACE_ENGINE` | `opencv_sface` | `opencv_sface`、`demo` 或 `deepface` |
| `OPENCV_MODEL_DIR` | `./data/models` | YuNet 和 SFace 模型目录 |
| `DEEPFACE_MODEL` | `Facenet512` | DeepFace 模型名称 |
| `DEEPFACE_DETECTOR` | `opencv` | DeepFace 检测器 |
| `MAX_UPLOAD_MB` | `10` | 单张图片大小限制 |
| `SEARCH_TOP_K` | `5` | 默认候选数量 |

更换人脸模型后，历史向量通常不兼容，需要清空并重新录入参考图。

## 数据规模升级

当前内存索引适合 MVP 和约十万张以内的参考图片。超过这一规模后，把 `VectorIndex` 替换成 Qdrant：

- collection 中保存归一化人脸向量；
- payload 保存 `person_id`、`image_id`；
- 使用 cosine distance；
- 查询 Top 100 图片后继续按人物聚合。

API 和前端不需要改变。

## 测试

```bash
pytest -q
```

## 许可证与数据提醒

本项目代码不附带任何明星图片、人脸库或模型权重。

- DeepFace 框架许可证不自动覆盖它加载的模型权重及训练数据；
- InsightFace 官方预训练模型通常仅限非商业研究，商用前需要单独确认；
- 录入参考照片前应核对照片许可证和人脸数据处理依据；
- 不要把 CelebA、VGGFace2 或网络抓取图片直接作为商业人物库。
