#!/bin/bash
cd "$(dirname "$0")/.."
python3 -m uvicorn api.main:app --reload --port 8420 --host 0.0.0.0
