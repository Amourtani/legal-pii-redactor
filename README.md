# Legal PII Redactor

中文法律文本脱敏工具，面向合同生成、案例展示、法律咨询记录和证据材料等场景。项目采用“规则引擎 + 法律 NER 小模型 + ONNX CPU 推理”的方式，目标是在本地 CPU 上实现秒级响应。


## 能做什么

- 规则识别：手机号、固定电话、身份证候选、邮箱、银行卡/银行账号、IP、URL、车牌、案号。
- NER 识别：当事人、证人、被害人、未成年人、代理人、律师、法官、地址、机构、律所、健康信息、商业秘密、合同项目。
- 脱敏输出：返回脱敏文本、命中实体、风险等级和是否需要人工复核。
- 训练闭环：用 OpenAI-compatible 强模型生成合成训练数据，本地训练 NER，导出 ONNX，INT8 量化后用 FastAPI 服务推理。

## 项目结构

```text
app/
  api.py              FastAPI 服务
  desensitizer.py     规则和 NER 编排
  masker.py           脱敏策略
  ner.py              ONNX NER 推理
  rules.py            规则识别
train/
  generate_synthetic.py  强模型生成训练数据
  validate_dataset.py    JSONL 数据校验
  train_ner.py           NER 训练
  export_onnx.py         导出 ONNX
  quantize_onnx.py       INT8 量化
config/
  llm.example.json       LLM 配置模板
data/examples/
  legal_ner.sample.jsonl 示例数据
tests/
```

## 0. 安装环境

推荐 Python 3.11。

Windows PowerShell:

```powershell
conda create --name legal-pii-redactor python=3.11
conda activate legal-pii-redactor

python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python -m pytest tests -q
```

Linux/macOS Bash:

```bash
conda create --name legal-pii-redactor python=3.11
conda activate legal-pii-redactor

python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python -m pytest tests -q
```

## 1. 先跑规则版服务

规则版不需要训练模型，适合先验证 API。

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Windows PowerShell 测试请求：

```powershell
$body = @{
  text = "原告张三，身份证号110101199001011234，电话13800138000，车牌浙A·B5678。"
  mode = "case_display"
  strict_level = "high"
} | ConvertTo-Json -Compress

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/desensitize" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Linux/macOS Bash 测试请求：

```bash
curl -X POST "http://127.0.0.1:8000/desensitize" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{"text":"原告张三，身份证号110101199001011234，电话13800138000，车牌浙A·B5678。","mode":"case_display","strict_level":"high"}'
```

## 2. 配置强模型

真实配置不能提交到 git。先复制模板。

Windows PowerShell:

```powershell
Copy-Item config/llm.example.json config/llm.json
```

Linux/macOS Bash:

```bash
cp config/llm.example.json config/llm.json
```

编辑 `config/llm.json`：

```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key": "replace-with-your-api-key",
  "model": "replace-with-your-strong-model",
  "timeout": 120
}
```

Windows PowerShell:

```powershell
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_API_KEY="your-key"
$env:LLM_MODEL="your-model"
```

Linux/macOS Bash:

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="your-key"
export LLM_MODEL="your-model"
```

## 3. 生成训练数据

生成阶段让强模型只输出实体文本，不输出 `start/end`。程序会自动定位 offset、校验实体格式，并跳过坏样本。只有整份文件校验通过后才会写入正式 JSONL。

强模型输出格式：

```json
{"text":"原告张三，电话13800138000。","entities":[{"text":"张三","label":"PERSON_PARTY"},{"text":"13800138000","label":"PHONE"}]}
```

最终训练格式：

```json
{"text":"原告张三，电话13800138000。","entities":[{"start":2,"end":4,"label":"PERSON_PARTY"},{"start":7,"end":18,"label":"PHONE"}]}
```

推荐第一轮生成 1 万到 1.5 万条。

Windows PowerShell:

```powershell
python -m train.generate_synthetic `
  --count 8000 `
  --batch-size 20 `
  --mode case_display `
  --output data/generated/legal_ner.train.jsonl `
  --max-batches 800

python -m train.generate_synthetic `
  --count 1000 `
  --batch-size 20 `
  --mode case_display `
  --output data/generated/legal_ner.dev.jsonl `
  --max-batches 120

python -m train.generate_synthetic `
  --count 1000 `
  --batch-size 20 `
  --mode case_display `
  --output data/generated/legal_ner.test.jsonl `
  --max-batches 120

python -m train.generate_synthetic `
  --count 5000 `
  --batch-size 20 `
  --mode contract_generation `
  --output data/generated/legal_ner.contract.jsonl `
  --max-batches 600
```

Linux/macOS Bash:

```bash
python -m train.generate_synthetic \
  --count 8000 \
  --batch-size 20 \
  --mode case_display \
  --output data/generated/legal_ner.train.jsonl \
  --max-batches 800

python -m train.generate_synthetic \
  --count 1000 \
  --batch-size 20 \
  --mode case_display \
  --output data/generated/legal_ner.dev.jsonl \
  --max-batches 120

python -m train.generate_synthetic \
  --count 1000 \
  --batch-size 20 \
  --mode case_display \
  --output data/generated/legal_ner.test.jsonl \
  --max-batches 120

python -m train.generate_synthetic \
  --count 5000 \
  --batch-size 20 \
  --mode contract_generation \
  --output data/generated/legal_ner.contract.jsonl \
  --max-batches 600
```

校验数据：

```bash
python -m train.validate_dataset data/generated/legal_ner.train.jsonl
python -m train.validate_dataset data/generated/legal_ner.dev.jsonl
python -m train.validate_dataset data/generated/legal_ner.test.jsonl
python -m train.validate_dataset data/generated/legal_ner.contract.jsonl
```

看到 `errors=0` 再训练。

## 4. 不同数据量的训练参数

| 数据量 | 目标 | 推荐 epochs | batch size | base model | 说明 |
|---:|---|---:|---:|---|---|
| 100-300 | 冒烟测试 | 3 | 16 | `uer/chinese_roberta_L-2_H-128` | 只验证流程，不看效果 |
| 1000-3000 | Demo | 8-12 | 16 | `uer/chinese_roberta_L-2_H-128` | 常见模板有一点效果 |
| 5000-10000 | 初步可用 | 10-15 | 16 | `uer/chinese_roberta_L-2_H-128` | 本地 CPU 友好，推荐起步 |
| 10000-50000 | 推荐训练 | 8-12 | 16/32 | `uer/chinese_roberta_L-2_H-128` | 合成数据覆盖更稳 |
| 50000+ | 进一步优化 | 5-10 | 32 | `hfl/chinese-macbert-base` 或小模型 | 有 GPU 时可换更强模型 |

如果模型训练后全预测 `O`，优先增加 epochs，例如从 3 提到 10 或 15。

## 5. 训练 NER

合并训练集。

Windows PowerShell:

```powershell
Get-Content data/generated/legal_ner.train.jsonl, data/generated/legal_ner.contract.jsonl `
  | Set-Content -Encoding utf8 data/generated/legal_ner.mixed.train.jsonl

python -m train.validate_dataset data/generated/legal_ner.mixed.train.jsonl
```

Linux/macOS Bash:

```bash
cat data/generated/legal_ner.train.jsonl data/generated/legal_ner.contract.jsonl \
  > data/generated/legal_ner.mixed.train.jsonl

python -m train.validate_dataset data/generated/legal_ner.mixed.train.jsonl
```

训练 CPU 友好的小模型。

Windows PowerShell:

```powershell
python -m train.train_ner `
  --train-file data/generated/legal_ner.mixed.train.jsonl `
  --dev-file data/generated/legal_ner.dev.jsonl `
  --base-model uer/chinese_roberta_L-2_H-128 `
  --output-dir models/legal-ner-v1 `
  --epochs 10 `
  --batch-size 16
```

Linux/macOS Bash:

```bash
python -m train.train_ner \
  --train-file data/generated/legal_ner.mixed.train.jsonl \
  --dev-file data/generated/legal_ner.dev.jsonl \
  --base-model uer/chinese_roberta_L-2_H-128 \
  --output-dir models/legal-ner-v1 \
  --epochs 10 \
  --batch-size 16
```

如果有 NVIDIA GPU，可尝试更强模型。

Windows PowerShell:

```powershell
python -m train.train_ner `
  --train-file data/generated/legal_ner.mixed.train.jsonl `
  --dev-file data/generated/legal_ner.dev.jsonl `
  --base-model hfl/chinese-macbert-base `
  --output-dir models/legal-ner-macbert `
  --epochs 5 `
  --batch-size 16
```

Linux/macOS Bash:

```bash
python -m train.train_ner \
  --train-file data/generated/legal_ner.mixed.train.jsonl \
  --dev-file data/generated/legal_ner.dev.jsonl \
  --base-model hfl/chinese-macbert-base \
  --output-dir models/legal-ner-macbert \
  --epochs 5 \
  --batch-size 16
```

## 6. 导出和量化

Windows PowerShell:

```powershell
python -m train.export_onnx `
  --model-dir models/legal-ner-v1 `
  --output models/legal-ner-v1-onnx

python -m train.quantize_onnx `
  --onnx-dir models/legal-ner-v1-onnx `
  --output-dir models/legal-ner-v1-onnx-int8
```

Linux/macOS Bash:

```bash
python -m train.export_onnx \
  --model-dir models/legal-ner-v1 \
  --output models/legal-ner-v1-onnx

python -m train.quantize_onnx \
  --onnx-dir models/legal-ner-v1-onnx \
  --output-dir models/legal-ner-v1-onnx-int8
```

导出时看到 `TracerWarning` 或量化时看到 preprocessing warning 通常不是失败。确认 `model.onnx` 存在即可。

## 7. 启用 NER 模型

Windows PowerShell:

```powershell
$env:NER_MODEL_DIR="models/legal-ner-v1-onnx-int8"
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Linux/macOS Bash:

```bash
export NER_MODEL_DIR="models/legal-ner-v1-onnx-int8"
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Windows PowerShell 测试：

```powershell
$body = @{
  text = "原告张三委托王律师处理案件，联系电话13800138000，住北京市朝阳区建国路88号。"
  mode = "case_display"
  strict_level = "high"
} | ConvertTo-Json -Compress

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/desensitize" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Linux/macOS Bash 测试：

```bash
curl -X POST "http://127.0.0.1:8000/desensitize" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{"text":"原告张三委托王律师处理案件，联系电话13800138000，住北京市朝阳区建国路88号。","mode":"case_display","strict_level":"high"}'
```

## 实体标签

```text
PERSON_PARTY, PERSON_VICTIM, PERSON_WITNESS, PERSON_MINOR, PERSON_AGENT,
LAWYER, JUDGE, ORG_PARTY, ORG_LAWFIRM, ADDRESS, ID_CARD, PHONE, EMAIL,
BANK_CARD, BANK_ACCOUNT, PLATE, HEALTH, BUSINESS_SECRET, CONTRACT_PROJECT
```


## 许可证

MIT License. 详见 [LICENSE](LICENSE)。
