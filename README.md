# PicQuery Models

为 PicQuery app 提供模型预处理、量化、压缩、ONNX 转换和导出。所有模型项目共用根目录的 `pyproject.toml`、`uv.lock` 和 `.venv`。

```text
picquery-models/
├── pyproject.toml / uv.lock / .python-version
├── export.sh
├── models/
│   ├── ml-mobileclip/
│   │   ├── export_mobileclip_models.py
│   │   ├── checkpoints/          # 原始 PyTorch 权重，不提交
│   │   ├── mobileclip/ / mobileclip2/
│   │   ├── third_party/open_clip/ # 本地 vendored OpenCLIP
│   │   └── docs/                 # 上游说明、示例和历史脚本
│   └── translation-zh-en/        # 留空，预留中译英模型
├── test/                         # 单元/集成测试与 benchmark
└── outputs/                      # 所有导出产物，不提交
```

安装与导出（需要 uv；Python 版本由 `.python-version` 指定）：

```bash
uv sync --locked
./export.sh                       # 默认 MobileCLIP2-S0
./export.sh s2
./export.sh s3
./export.sh s0 --checkpoint /path/to/mobileclip2_s0.pt
uv run --locked pytest -q -s                 # 默认 S0
uv run --locked pytest -q -s --model s2
uv run --locked pytest -q -s --model s3
uv run --locked python test/benchmark_visual_onnx.py
```

`export.sh` 可从任何工作目录执行。默认优先加载 `models/ml-mobileclip/checkpoints/mobileclip2_s0.pt`；没有本地 checkpoint 时由 OpenCLIP 下载官方权重。显式 `--checkpoint` 不存在时直接报错。测试必须使用与导出相同的原始权重，可通过 `pytest --checkpoint PATH --outputs PATH` 指定，不会静默跳过。

| 产物 | 权重 | 输入 | 输出 |
|---|---|---|---|
| `outputs/mobileclip2_s0_text.onnx` | per-channel dynamic INT8（矩阵和 embedding；少量 norm/bias 保持 FP32） | INT64 `[batch,77]` token IDs | FP32 `[batch,512]` |
| `outputs/mobileclip2_s0_visual.onnx` | FP16 | FP32 `[batch,3,256,256]` | FP32 `[batch,512]` |

接口以 PicQuery 的 `lib/src/engine/ort_engine.dart` 为准：text 用 `Int64List`，visual 用 `Float32List`。S0/S2 图像为 RGB/CHW、范围 `[0,1]`，不额外套用 ImageNet mean/std。输出是未归一化 embedding；app 会进行 L2 归一化。tokenizer 使用 CLIP BPE、SOT=49406、EOT=49407，固定 77 tokens。

Visual 通过 ONNX Runtime 的 [FP16 转换](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)保留 FP32 I/O；text 使用 [动态量化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。转换中间文件放在 `outputs/` 的临时目录，成功或失败后清理。先校验两份模型和 batch=1/2 推理，再替换最终文件。

测试对比重参数化前的原始 FP32 PyTorch 模型，覆盖真实示例图、黑/灰/白图、随机图，以及普通、空、数字和超长文本。验证 ONNX 接口、压缩参数占比、单条/批量推理、有限值、embedding cosine 和相对 L2 误差，以及图文检索相似度。阈值：text cosine ≥ 0.99、相对 L2 ≤ 0.15；S0/S2 visual cosine ≥ 0.9998、相对 L2 ≤ 0.02；更大的 S3 visual cosine ≥ 0.9995、相对 L2 ≤ 0.03；图文 cosine-score 最大误差 ≤ 0.025。这是回归样本测试，不能替代完整数据集的检索评估。

导出入口仅支持 MobileCLIP2 的 `s0`、`s2`、`s3`，文件名统一为 `mobileclip2_<型号>_text.onnx` 和 `mobileclip2_<型号>_visual.onnx`。S0/S2 输出 512 维，S3 输出 768 维；S3 图像输入还需要官方 CLIP mean/std 归一化（mean = 0.48145466, 0.4578275, 0.40821073；std = 0.26862954, 0.26130258, 0.27577711）。现有 PicQuery 的 512 维与 `[0,1]` 预处理接口适用于 S0/S2，接入 S3 时需要同步调整 app。

<!-- 翻译模型未来统一命名：outputs/mt_zho-eng.fp32.quantized.onnx；目前不实现下载、导出或推理。 -->

MobileCLIP 与 vendored OpenCLIP 保留原始 LICENSE、LICENSE_DATA、LICENSE_MODELS 和 ACKNOWLEDGEMENTS，使用权重前请参阅对应许可。

迁移采用复制并保留相邻原目录作为备份；原目录的 `.git`、`.venv` 和缓存不作为本项目的一部分。历史 ONNX 文件位于 `outputs/legacy/`，S0/S2/S3 原始权重位于 `models/ml-mobileclip/checkpoints/`。

本次 S0 CPU 验证：9 项测试全部通过。text / visual 最低 embedding cosine 分别为 0.999367 / 0.999837，最大相对 L2 误差分别为 3.59% / 1.81%，图文 cosine-score 最大差值为 0.003663。最终模型体积分别为 61.61 / 21.83 MiB。测试与 benchmark 日志保存在 `outputs/test.log` 和 `outputs/benchmark.log`。

MobileCLIP2 型号验证（CPU 回归样本）：

| 型号 | text / visual 体积 MiB | text / visual 最低 cosine | 最大图文相似度差值 |
|---|---|---|---|
| S0 | 61.61 / 21.83 | 0.999367 / 0.999837 | 0.003663 |
| S2 | 61.61 / 68.42 | 0.999318 / 0.999975 | 0.003157 |
| S3 | 119.35 / 238.92 | 0.997920 / 0.999795 | 0.005111 |

分型号日志位于 `outputs/export_s2.log`、`outputs/export_s3.log` 和 `outputs/test_s0.log`、`outputs/test_s2.log`、`outputs/test_s3.log`。
