# Contributing

Thanks for helping improve Legal PII Redactor.

## Development

```powershell
conda create --name legal-pii-redactor python=3.11
conda activate legal-pii-redactor
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python -m pytest tests -q
```

## Pull Request Checklist

- Do not commit `config/llm.json`, generated data, trained models, or API keys.
- Add tests for behavior changes.
- Run `python -m pytest tests -q` before opening a PR.
- For training changes, validate JSONL files with `python -m train.validate_dataset <file>`.

## Data Safety

Do not submit real client contracts, real case materials, or unredacted personal information in issues, tests, examples, or pull requests.
