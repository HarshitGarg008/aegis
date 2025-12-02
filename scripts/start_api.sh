#!/usr/bin/env bash
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
