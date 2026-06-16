# Configuration

Copy `llm.example.json` to `llm.json` before generating synthetic data:

```powershell
Copy-Item config/llm.example.json config/llm.json
```

`config/llm.json` is ignored by git because it contains private API keys.

`timeout` is the local HTTP read timeout in seconds. If a request still fails
earlier than this value, the upstream API gateway or proxy may have a shorter
limit; reduce `--batch-size` or use another endpoint.
