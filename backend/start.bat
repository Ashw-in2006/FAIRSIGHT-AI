@echo off
setlocal

set FAIRSIGHT_PORT=8001

if not exist venv (
  python -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python main.py
