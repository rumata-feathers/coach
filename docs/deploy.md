# Deploying Career Coach to Railway

This document covers the first production deploy: a Railway service backed by Supabase Postgres.
No auth, no frontend — just the FastAPI backend accessible over the internet.

---

## Prerequisites

- A [Railway](https://railway.app) account with a project created
- A [Supabase](https://supabase.com) project with the pgvector extension enabled
- The repo pushed to GitHub (Railway connects via GitHub)
- Docker installed locally (for the local smoke test)

---

## Required environment variables

Set these in Railway's **Variables** panel for your service.

| Variable | Description | Example |
|---|---|---|
| `SUPABASE_DB_URL` | Supabase **Postgres connection string** (use the *pooler* URL for Railway) | `postgresql://postgres.xxxx:password@aws-0-eu-west-2.pooler.supabase.com:6543/postgres` |
| `HUGGINGFACE_API_TOKEN` | HuggingFace Inference API token (required for all LLM calls) | `hf_…` |
| `TAVILY_API_KEY` | Tavily web search key (required for Flow C / Researcher) | `tvly-…` |
| `ANTHROPIC_API_KEY` | Optional fallback if you add Anthropic models later | `sk-ant-…` |
| `EMBEDDING_DIM` | Must match the vector dimension in your DB | `384` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

Railway sets these automatically — **do not add them manually**:

| Variable | Set by | Notes |
|---|---|---|
| `PORT` | Railway | The port uvicorn binds to. Do not hardcode. |
| `GIT_SHA` | Railway | The commit SHA being deployed. Appears in `/health`. |

> **Supabase pooler URL vs direct URL**
> Use the *Session mode* pooler URL (port 6543) from the Supabase dashboard
> under **Settings → Database → Connection string → URI**.  The Transaction mode
> pooler (port 5432) drops prepared statements that asyncpg needs.

---

## Step-by-step: first deploy

### 1. Enable pgvector in Supabase

In the Supabase SQL editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

### 2. Connect the repo to Railway

1. In your Railway project, click **New Service → GitHub Repo**.
2. Select the `career-coach` repository and the `main` branch.
3. Railway detects the `Dockerfile` automatically.

### 3. Set environment variables

In the service's **Variables** tab, add the variables from the table above.
Paste each value, press Enter. Railway stores them encrypted.

### 4. Deploy

Click **Deploy** (or push a commit — Railway auto-deploys on push to `main`).

Railway will:
1. Build the Docker image from `Dockerfile`.
2. Run the `releaseCommand` from `railway.toml`:
   ```
   uv run python scripts/run_migrations_prod.py
   ```
   This applies any pending SQL migrations to your Supabase database.
3. Start the app with uvicorn on `$PORT`.
4. Poll `GET /health` until it returns 200, then cut over traffic.

### 5. Verify the deploy

Once the service shows **Active** in Railway, copy the public URL and run:

```bash
# Should return {"status":"ok","version":"0.0.1","git_sha":"<sha>"}
curl https://<your-service>.railway.app/health

# Create a test user
curl -X POST https://<your-service>.railway.app/users \
  -H 'Content-Type: application/json' \
  -d '{"display_name": "test-friend"}'

# Chat (use the user_id from the previous response)
curl -X POST https://<your-service>.railway.app/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "<uuid>", "message": "I am finishing my maths degree — what should I do next?"}'
```

---

## Local smoke test (before pushing to Railway)

Build and run the image exactly as Railway will:

```bash
# Build
docker build -t career-coach .

# Run (reads SUPABASE_DB_URL and API keys from your local .env)
docker run -p 8000:8000 --env-file .env career-coach

# In another terminal
curl http://localhost:8000/health
```

> The local run requires `SUPABASE_DB_URL` in `.env` to point at a reachable
> Postgres (your local docker-compose or Supabase directly).  The app will
> refuse to start if the DB is unreachable.

---

## Running migrations manually

If you need to apply migrations outside of a deploy (e.g. against a staging DB):

```bash
# Against Supabase directly
SUPABASE_DB_URL="postgresql://..." uv run python scripts/run_migrations_prod.py

# Against local docker-compose (add ALLOW_LOCAL_DB to bypass the safety guard)
ALLOW_LOCAL_DB=1 uv run python scripts/run_migrations_prod.py
```

Migrations are idempotent — safe to run multiple times.  Already-applied
versions are skipped (tracked in the `schema_migrations` table).

---

## How to roll back

Railway keeps a full deploy history. To revert to a previous version:

1. Open your service in the Railway dashboard.
2. Click **Deployments** in the left sidebar.
3. Find the last known-good deploy and click the **⋯** menu → **Redeploy**.

The `releaseCommand` runs again on rollback, but since migrations are forward-only
and idempotent, this is safe.  If a migration needs to be reversed, write a new
migration (e.g. `005_rollback_x.sql`) and redeploy from `main`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Deploy fails at `releaseCommand` | DB unreachable or `SUPABASE_DB_URL` wrong | Check the variable in Railway → verify the Supabase pooler URL |
| `/health` returns 500 | DB pool failed to connect at startup | Same as above — check DB URL and network access |
| `Researcher timed out` in logs | `TAVILY_API_KEY` not set | Add the key in Railway Variables |
| `BadRequestError: json_validate_failed` | HF model overloaded | Transient — retry or check HF API status |
| Container restarts repeatedly | App crashes at startup | Check deploy logs in Railway → **Logs** tab |

---

## Architecture notes

- **No persistent local state.** The container is stateless; all state lives in Supabase.
- **Background tasks** (Profiler, Distillation) run as `asyncio` tasks inside the
  same process.  For high traffic, move these to Railway's background worker
  service (separate service, same image, different `startCommand`).
- **The `kb/` directory** is baked into the image. KB updates require a redeploy.
- **Port binding.** Railway sets `$PORT` at runtime; the `Dockerfile` CMD uses
  `${PORT:-8000}` so `docker run -p 8000:8000` works without `-e PORT=8000`.
