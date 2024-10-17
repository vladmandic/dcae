#!/usr/bin/env bash
python -m venv venv
source venv/bin/activate
pip install uv
uv pip install --requirement requirements.txt
