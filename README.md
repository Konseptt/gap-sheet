# Gap Sheet

A small **Flask** app that compares a **job posting** to a **résumé** (pasted text or **PDF**), calls **NVIDIA NIM** (`meta/llama-4-maverick-17b-128e-instruct`), and returns structured JSON: missing skills, keywords to strengthen, and prioritized fixes.

Uploads are processed in memory only; nothing is written to disk for persistence.

## Requirements

- **Python 3.12** (see `.python-version`)
- An **NVIDIA API key** with access to the integrate API used in `app.py`

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your secrets (see below). **Do not commit `.env`.**

## Environment variables

| Variable | Required | Notes |
|----------|----------|--------|
| `NVIDIA_API_KEY` | Yes (for analysis) | Used to call NVIDIA chat completions. |
| `SECRET_KEY` | **Yes on Vercel** | Stable random string for sessions and CSRF. The app refuses to start on Vercel without it. Locally, a random key is generated if unset. |
| `FLASK_DEBUG` | No | Set to `1` **only** for local debugging. Never enable in production. |
| `SESSION_COOKIE_SECURE` | No | On Vercel, secure cookies are enabled automatically when `VERCEL=1`. |

On **Vercel**, set `NVIDIA_API_KEY` and `SECRET_KEY` under **Project → Settings → Environment Variables**. The platform typically sets `VERCEL`; you normally do not need to add it yourself.

## Run locally

```bash
flask --app app run
```

Or with gunicorn (closer to production):

```bash
gunicorn -w 1 -b 127.0.0.1:5000 app:app
```

Open `http://127.0.0.1:5000`. The UI posts to `POST /api/analyze` with CSRF protection (`X-CSRF-Token` and session cookie).

## Deploy on Vercel

1. Connect this repo to Vercel (Python runtime is detected from `requirements.txt` / layout).
2. Set **`NVIDIA_API_KEY`** and **`SECRET_KEY`** in the project environment.
3. Optional: `.vercelignore` excludes local env files and dev-only scripts from the upload bundle.

Static assets live under `public/static/` and are served at `/static/...` per Flask configuration.

## Security notes

- Never commit real API keys or `.env`.
- Prefer strong, long `SECRET_KEY` values in production and keep them stable across deploys so sessions remain valid.
- The app applies security headers (CSP, frame denial, etc.) and rate limits sensitive routes; see `app.py` for details.

## License

Use and modify as you like for your own projects; there is no warranty.
