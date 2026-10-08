# MobileCLIP 模型项目

为 PicQuery 导出 text INT8 与 visual FP16 ONNX，保持 app 实际接口。依赖和 uv 环境由仓库根目录管理。

从根目录运行 `./export.sh`；完整操作和精度阈值见根目录 README。

- `export_mobileclip_models.py`：唯一当前导出入口，仅支持 MobileCLIP2 S0/S2/S3，默认 S0。
- `checkpoints/`：本地原始权重；未找到时使用官方预训练标签。
- `mobileclip/`、`mobileclip2/`、`third_party/open_clip/`：模型实现与 vendored 依赖。
- `docs/UPSTREAM_README.md`：原项目说明；历史脚本带 `.legacy` 标记，仅供参考。
- `training/`、`eval/`、`ios_app/`、`results/`：保留上游资源和原始参考数据。
- 所有导出的 ONNX 文件统一保存到根目录 `outputs/`。
