@echo off
cd /d "%~dp0"
python job_dashboard.py
if errorlevel 1 pause
