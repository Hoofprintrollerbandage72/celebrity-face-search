# MVP 闭环验收

验收日期：2026-07-19（Asia/Shanghai）

## 闭环范围

- 真实人脸检测：OpenCV YuNet；
- 人脸对齐和特征：OpenCV SFace；
- 人物管理：创建人物、上传参考图、图片源一键下载关联、删除人物；
- 数据持久化：SQLite、参考图文件、float32 归一化向量；
- 检索：余弦相似度 Top-K，按人物聚合多张参考图；
- 用户界面：上传预览、返回数量、候选人物和参考图；
- 批量能力：目录导入脚本；
- 可重复演示：带来源和许可证信息的 Commons 图片。
- 来源追溯：检索结果返回最佳参考图的图片直链、来源页面和许可证记录。

## 图片源快速配置验收

闭环入口位于“人物库 → 图片源快速配置”。一次请求可创建或选择人物，并处理
最多 20 个公开图片源。每张图片独立返回成功、重复跳过或失败，不会因一张坏图
中断整批配置。

安全和完整性检查包括：

- 仅允许 HTTP(S)，禁止 URL 凭据和非常用端口；
- 每次请求及每次重定向都解析地址，拦截本机、内网和保留 IP；
- 限制响应大小、跳转次数和超时时间；
- 用 Pillow 验证真实图片格式，用人脸引擎验证恰好一张人脸；
- 图片保存到 `data/references/source-imports/<person_id>/`，SQLite 同步记录来源；
- `(person_id, source_url)` 唯一索引保证重复操作幂等。

## 模型

| 文件 | SHA-256 | 目录许可证 |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT |
| `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |

模型由 `scripts/setup_models.py` 从 OpenCV 官方 Hugging Face 仓库下载，并在写入后校验哈希。

## 真实图片验收

人物库：

- Barack Obama：2009 官方肖像，CC BY 3.0，Pete Souza；
- Donald Trump：2017 官方肖像，美国联邦政府作品，Public Domain。

查询图：

- Donald Trump：2025 官方头像，与入库的 2017 照片不是同一张图片。

实测结果：

```text
Top-1: Donald Trump / cosine=0.5678 / aggregate=0.5678
REAL_MVP_SMOKE_TEST=PASS
```

这项测试覆盖：下载图片、创建人物、参考图人脸检测、向量写入、查询图人脸检测、特征提取、Top-K 检索和 Top-1 身份验证。

## 自动化验证

```text
pytest: 8 passed
JavaScript syntax: pass
Python compileall: pass
Docker Compose config: pass
GET /: 200
GET /api/health: 200
GET /openapi.json: 200
```

## 复现

终端一：

```bash
./scripts/start_mvp.sh
```

终端二：

```bash
source .venv/bin/activate
python scripts/seed_demo_library.py
```

成功标志为 `REAL_MVP_SMOKE_TEST=PASS`。

## 边界

- 相似度是模型距离，不是身份概率；
- 演示库只有两个人，用于验证闭环，不代表正式公众人物覆盖率；
- 正式上线前仍需扩充合法来源人物库并完成不同人群、年龄、侧脸、妆容和 AI 生成图的阈值测试。
