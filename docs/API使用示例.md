# API 使用示例

默认服务地址为 `http://127.0.0.1:8000`，交互式 OpenAPI 文档位于 `/docs`。典型输入是待发布的 AI 短剧角色图；API 返回公众人物相似候选和来源线索，不返回侵权结论。

## 健康检查

```bash
curl http://127.0.0.1:8000/api/health
```

响应示例：

```json
{
  "status": "ok",
  "engine": "opencv-sface",
  "engine_available": true,
  "real_face_recognition": true,
  "reference_images": 100,
  "persons": 20
}
```

## 创建和查询人物

```bash
curl -X POST http://127.0.0.1:8000/api/library/persons \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Example Person",
    "external_id": "wikidata:Q0000",
    "aliases": ["Example Alias"]
  }'
```

```bash
curl 'http://127.0.0.1:8000/api/library/persons?limit=50&q=Example'
```

`external_id` 非空时必须唯一，适合保存 Wikidata QID 或内部身份 ID。

## 手工上传参考图

```bash
curl -X POST http://127.0.0.1:8000/api/library/persons/PERSON_ID/images \
  -F 'image=@./reference.jpg' \
  -F 'source_url=https://images.example.org/reference.jpg' \
  -F 'source_page_url=https://example.org/photo-page' \
  -F 'license_code=CC BY 4.0'
```

真实人脸引擎要求参考图恰好包含一张可检测人脸，否则返回 `422`。

## 图片源一键配置

选择已有人物：

```bash
curl -X POST http://127.0.0.1:8000/api/library/quick-source-import \
  -H 'Content-Type: application/json' \
  -d '{
    "person_id": "PERSON_ID",
    "sources": [
      {
        "image_url": "https://images.example.org/portrait-01.jpg",
        "source_page_url": "https://example.org/photo-01",
        "license_code": "CC BY 4.0"
      },
      {
        "image_url": "https://images.example.org/portrait-02.jpg",
        "source_page_url": "https://example.org/photo-02",
        "license_code": "CC BY 4.0"
      }
    ]
  }'
```

同时创建新人物：

```bash
curl -X POST http://127.0.0.1:8000/api/library/quick-source-import \
  -H 'Content-Type: application/json' \
  -d '{
    "person": {
      "name": "Example Person",
      "external_id": "example:person",
      "aliases": []
    },
    "sources": [{
      "image_url": "https://images.example.org/portrait.jpg",
      "source_page_url": "https://example.org/photo-page",
      "license_code": "Permission recorded internally"
    }]
  }'
```

响应中的 `summary` 会分别统计 `imported`、`skipped` 和 `failed`。单张失败不会回滚其他成功图片；相同人物与相同 `image_url` 重复提交时会幂等跳过。

读取图片源填写规则：

```bash
curl http://127.0.0.1:8000/api/library/source-guide
```

## 上传 AI 角色查询图

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -F 'image=@./query.jpg' \
  -F 'top_k=5'
```

一个查询图可以检测出多张人脸。每张脸独立返回：

- `box`：检测框；
- `detection_confidence`：人脸检测置信度；
- `candidates`：按人物聚合的候选列表；
- `similarity`：最佳单张参考图余弦相似度；
- `aggregate_similarity`：最佳分数与 Top-3 均值的加权聚合；
- `source_url`、`source_page_url`、`license_code`：最佳参考图追溯信息。

业务层不要把 `aggregate_similarity` 直接显示为“侵权概率”。阈值应使用目标画风和实际生成模型的验证集完成标定，详细方法见 [`AI短剧角色图风险筛查指南.md`](AI短剧角色图风险筛查指南.md)。

## 人物库维护

```bash
curl http://127.0.0.1:8000/api/library/stats
curl -X POST http://127.0.0.1:8000/api/library/reload
curl -X DELETE http://127.0.0.1:8000/api/library/persons/PERSON_ID
```

删除人物会同时删除其数据库参考记录和本地参考图片，并重载内存索引。

## 常见错误

| 状态码 | 场景 |
|---|---|
| `400` | 空文件、无效图片格式 |
| `404` | 人物或参考图片不存在 |
| `409` | 人物库为空、外部 ID 重复或向量维度不一致 |
| `413` | 图片超过大小限制 |
| `422` | 未检测到单一参考人脸、参数校验失败 |
