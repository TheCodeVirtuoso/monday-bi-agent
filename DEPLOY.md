# Deploying

The app is a single FastAPI service with no database and no build step. It
runs against the workbooks in `data/` by default, so it deploys and works
without any monday.com credentials.

**Only one secret is strictly required: an LLM API key.**

---

## Render (recommended — free tier, blueprint included)

1. Push this repo to GitHub.
2. Render → **New → Blueprint** → select the repo. It reads `render.yaml`.
3. When prompted, paste `GROQ_API_KEY`. Leave the monday.com values blank.
4. Deploy. Render polls `/api/health`, which performs a real board load — so
   a green check means the data path works, not just that the process is up.

## Railway / Fly / Cloud Run

A `Dockerfile` is included and honours `$PORT`:

```bash
docker build -t monday-bi-agent .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... monday-bi-agent
```

## Anything Procfile-based

```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | **yes** | Or `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` |
| `LLM_PROVIDER` | no | `groq` (default), `openrouter`, `anthropic`. Auto-detects from whichever key is set. |
| `GROQ_MODEL` | no | Default `openai/gpt-oss-120b` |
| `USE_MOCK_DATA` | no | `true` (default) reads `data/`. Set `false` for live boards. |
| `MONDAY_API_TOKEN` | no | Required only when `USE_MOCK_DATA=false` |
| `MONDAY_DEALS_BOARD_ID` | no | ″ |
| `MONDAY_WORK_ORDERS_BOARD_ID` | no | ″ |

### A note on Groq model ids

Groq has **retired the `llama-3.x` ids** that most tutorials still reference —
they now return 404. Verified working on this account:

| Model | Tool calling |
|---|---|
| `openai/gpt-oss-120b` | ✅ default |
| `openai/gpt-oss-20b` | ✅ smaller, faster |
| `qwen/qwen3.8-27b` | ✅ |
| `qwen/qwen3.6-27b` | ❌ does not reliably call tools |

If you see *"does not recognise the model"*, list your account's live
catalogue:

```bash
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     https://api.groq.com/openai/v1/models
```

---

## Post-deploy check

```bash
curl https://<your-app>/api/health
```

Expect `"status": "ok"`, `rows_usable` of **342** deals and **176** work
orders, and `"key_present": true` under `llm`. Then open the app in a private
window and ask *"How much cash is outstanding?"* — it should return
₹3.63 Cr and name `WOCOMPANY_010` as the largest debtor.

## Security

`.env` is gitignored and `.dockerignore`d. Never commit a real key — set
secrets in the host's dashboard. Rotate any key that has been pasted into a
chat, an issue, or a screen share.
