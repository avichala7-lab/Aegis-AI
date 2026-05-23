#!/usr/bin/env bash
set -e

if [ -d "venv" ]; then
  source venv/bin/activate
fi

streamlit run app.py
