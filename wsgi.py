"""WSGI entry for production servers (e.g. gunicorn wsgi:app)."""

from app import app

__all__ = ["app"]
