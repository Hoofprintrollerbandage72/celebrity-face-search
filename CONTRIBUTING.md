# Contributing

感谢关注公众人物人脸检索项目。提交代码前，请先确认改动不会把真实人脸数据、模型权重、数据库或凭据加入仓库。

## 本地开发

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
FACE_ENGINE=demo uvicorn app.main:app --reload
```

真实人脸引擎可运行 `python3 scripts/setup_models.py` 后设置 `FACE_ENGINE=opencv_sface`。

## 提交前检查

```bash
pytest -q
python3 -m compileall -q app scripts tests
node --check app/static/app.js
docker compose config --quiet
```

Pull Request 请说明：

- 改动解决的问题；
- API、数据库或向量兼容性影响；
- 已运行的测试；
- 是否涉及模型、数据集、图片来源或许可证变化；
- 涉及识别阈值时使用的验证集、指标和误报率。

## 数据与隐私

- 不要在 Issue、测试夹具或 PR 中上传真实人物照片、人脸向量或私人身份信息；
- 测试优先使用程序生成的纯色图片和 `demo` 引擎；
- 不要提交 `.env`、Token、Cookie、数据库、模型文件或下载断点；
- 新增网络下载功能必须覆盖 SSRF、重定向、大小限制和内容验证测试。

## 许可证

仓库目前尚未选择代码开源许可证。外部贡献在仓库所有者明确许可证和贡献条款前可能无法合并。

