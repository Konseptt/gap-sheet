"""
Gap finder: job posting vs resume — proxies to NVIDIA NIM chat completions.
Secrets: NVIDIA_API_KEY; SECRET_KEY (recommended; on Vercel a derived key is used if unset). No disk storage of uploads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent

# Load .env from project directory (local dev)
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from openai import OpenAI
from flask import Flask, jsonify, render_template, request, session
from pypdf import PdfReader
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix


def _flask_secret_key() -> str:
    """Session signing key: prefer SECRET_KEY env; on Vercel derive a stable per-deployment value."""
    raw = (os.environ.get("SECRET_KEY") or "").strip()
    if raw:
        return raw
    if os.environ.get("VERCEL") == "1":
        # Same value for every instance in this deployment (required for signed cookies).
        # Set SECRET_KEY in the dashboard to control rotation and avoid relying on deployment ids.
        parts = [
            os.environ.get("VERCEL_PROJECT_ID", ""),
            os.environ.get("VERCEL_DEPLOYMENT_ID", ""),
            os.environ.get("VERCEL_GIT_COMMIT_SHA", ""),
        ]
        if any(parts):
            h = hashlib.sha256()
            h.update(b"gap-sheet-flask-session-v1\x00")
            h.update("|".join(parts).encode("utf-8"))
            return h.hexdigest()
    return secrets.token_hex(32)


# Vercel: serve assets from public/static (CDN + Flask); see Vercel Flask docs.
_STATIC = _ROOT / "public" / "static"
app = Flask(__name__, static_folder=str(_STATIC), static_url_path="/static")

if os.environ.get("VERCEL") == "1":
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=1,
        x_proto=1,
        x_host=1,
        x_port=1,
        x_prefix=1,
    )

MAX_PDF_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 50
MAX_EXTRACT_CHARS = 120_000
MAX_JOB_CHARS = 100_000
MAX_RESUME_TEXT_CHARS = 100_000

app.config["MAX_CONTENT_LENGTH"] = MAX_PDF_BYTES + 256 * 1024
app.config["SECRET_KEY"] = _flask_secret_key()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
_on_vercel = os.environ.get("VERCEL") == "1"
app.config["SESSION_COOKIE_SECURE"] = _on_vercel or os.environ.get("SESSION_COOKIE_SECURE", "").lower() in (
    "1",
    "true",
    "yes",
)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=[],
)

MODEL = "moonshotai/kimi-k3"


def nvidia_api_key() -> str:
    return (os.environ.get("NVIDIA_API_KEY") or "").strip()


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-site"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com data:; "
        "img-src 'self' data: blob:; "
        "connect-src 'self';"
    )
    return resp


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Upload too large."}), 413


@app.errorhandler(429)
def rate_limited(_e):
    return jsonify({"error": "Too many requests. Wait a moment and try again."}), 429


def build_system_prompt() -> str:
    return (
        "You compare a job posting to a candidate resume. Be blunt and specific. "
        "Output ONLY valid JSON, no markdown fences, no commentary. Schema:\n"
        '{"missing_skills":[{"skill":"string","note":"short why it matters from the posting"}],'
        '"missing_keywords":[{"phrase":"exact phrase from posting","gap":"what resume lacks"}],'
        '"priority_fixes":[{"rank":1,"action":"concrete edit or skill to add","why":"one line"}]}\n'
        "Rank priority_fixes 1 = do first. If posting or resume empty, use empty arrays."
    )


def _fence_body(text: str) -> str | None:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return m.group(1).strip() if m else None


def _balanced_json_object(text: str) -> dict[str, Any] | None:
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        pos = start + 1
                        break
                    if isinstance(obj, dict):
                        return obj
                    pos = i + 1
                    break
        else:
            pos = start + 1
    return None


def extract_json(text: str) -> dict[str, Any] | None:
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    candidates: list[str] = [text]
    fb = _fence_body(text)
    if fb:
        candidates.insert(0, fb)
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        balanced = _balanced_json_object(cand)
        if balanced is not None:
            return balanced
    return None


def _usable_reply(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not set(stripped) <= {"!"}


def message_content(message: dict[str, Any]) -> str:
    raw = message.get("content")
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and _usable_reply(reasoning):
        if raw is None or (isinstance(raw, str) and not _usable_reply(raw)):
            return reasoning
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(raw)


def sanitize_body_text(s: str, max_len: int) -> str:
    if not s:
        return ""
    s = s.replace("\x00", "")
    if len(s) > max_len:
        s = s[:max_len]
    return s.strip()


def extract_pdf_text(raw: bytes) -> str:
    if len(raw) > MAX_PDF_BYTES:
        raise ValueError("pdf_too_large")
    if not raw.startswith(b"%PDF"):
        raise ValueError("not_pdf")
    try:
        reader = PdfReader(BytesIO(raw), strict=False)
    except Exception:
        raise ValueError("bad_pdf") from None
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except Exception:
            ok = 0
        if ok == 0:
            raise ValueError("encrypted_pdf")
    n = len(reader.pages)
    if n > MAX_PDF_PAGES:
        raise ValueError("too_many_pages")
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if total + len(t) > MAX_EXTRACT_CHARS:
            parts.append(t[: max(0, MAX_EXTRACT_CHARS - total)])
            break
        parts.append(t)
        total += len(t)
        if total >= MAX_EXTRACT_CHARS:
            break
    out = "\n\n".join(parts)
    out = sanitize_body_text(out, MAX_EXTRACT_CHARS)
    if not out:
        raise ValueError("no_text")
    return out


def require_csrf() -> tuple[None, None] | tuple[Any, int]:
    expected = session.get("csrf")
    got = request.headers.get("X-CSRF-Token", "")
    if not expected or not got:
        return jsonify({"error": "Session expired or invalid. Refresh the page."}), 403
    try:
        ok = secrets.compare_digest(str(got), str(expected))
    except (ValueError, TypeError):
        ok = False
    if not ok:
        return jsonify({"error": "Session expired or invalid. Refresh the page."}), 403
    return None, None


def parse_analyze_input() -> tuple[str | None, str | None, tuple[Any, int] | None]:
    posting = ""
    resume = ""

    ct = (request.content_type or "").lower()
    if "multipart/form-data" in ct:
        posting = sanitize_body_text(request.form.get("job_posting", ""), MAX_JOB_CHARS)
        resume = sanitize_body_text(request.form.get("resume", ""), MAX_RESUME_TEXT_CHARS)
        up = request.files.get("resume_pdf")
        if up and up.filename:
            raw = up.read(MAX_PDF_BYTES + 1)
            if len(raw) > MAX_PDF_BYTES:
                return None, None, (jsonify({"error": "PDF exceeds size limit."}), 413)
            try:
                resume = extract_pdf_text(raw)
            except ValueError as e:
                code = str(e)
                if code == "encrypted_pdf":
                    msg = "That PDF is password-protected. Remove the lock or paste text instead."
                elif code in ("not_pdf", "bad_pdf"):
                    msg = "That file doesn’t look like a readable PDF."
                elif code in ("too_many_pages",):
                    msg = "That PDF has too many pages."
                elif code in ("no_text",):
                    msg = "No readable text was found in that PDF."
                else:
                    msg = "Couldn’t read that PDF."
                return None, None, (jsonify({"error": msg}), 400)
    else:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            data = {}
        posting = sanitize_body_text((data.get("job_posting") or ""), MAX_JOB_CHARS)
        resume = sanitize_body_text((data.get("resume") or ""), MAX_RESUME_TEXT_CHARS)

    if not posting and not resume:
        return None, None, (jsonify({"error": "Add a job posting and/or résumé (text or PDF)."}), 400)

    return posting, resume, None


@app.route("/")
def index():
    session["csrf"] = secrets.token_hex(16)
    return render_template(
        "index.html",
        dateline=datetime.now().strftime("%A · %B %d, %Y"),
        csrf_token=session["csrf"],
        has_nvidia_key=bool(nvidia_api_key()),
    )


@app.route("/api/analyze", methods=["POST"])
@limiter.limit("30 per minute")
def analyze():
    err = require_csrf()
    if err[0] is not None:
        return err[0], err[1]

    posting, resume, bad = parse_analyze_input()
    if bad is not None:
        return bad[0], bad[1]

    assert posting is not None and resume is not None

    api_key = nvidia_api_key()
    if not api_key:
        return jsonify(
            {
                "error": "No API key configured.",
                "detail": "Set NVIDIA_API_KEY in your environment, or add it to a .env file in this folder (see .env.example).",
            }
        ), 503

    user_content = (
        "--- JOB POSTING ---\n"
        f"{posting}\n\n"
        "--- RESUME ---\n"
        f"{resume}\n"
    )

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=120,
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            temperature=1,
            top_p=0.95,
            max_tokens=1024,
            stream=False,
            extra_body={"reasoning_effort": "low"},
        )
        raw = message_content(completion.choices[0].message.model_dump())
    except Exception:
        return jsonify({"error": "Could not reach the model service."}), 502

    if not (raw or "").strip():
        return jsonify({"error": "Empty reply from model."}), 502

    parsed = extract_json(raw)
    if parsed is not None and not isinstance(parsed, dict):
        parsed = None
    return jsonify(
        {
            "parsed": parsed,
            "raw_text": raw if parsed is None else None,
        }
    )


def _flask_debug() -> bool:
    return os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    debug = _flask_debug()
    print(f"Gap Sheet web app → http://127.0.0.1:{port}/ (debug={'on' if debug else 'off'})")
    app.run(host="127.0.0.1", port=port, debug=debug)
