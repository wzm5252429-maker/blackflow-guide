@echo off
cd /d "%~dp0.."
py -3.13 -m blackflow_live --open-site
if errorlevel 1 pause
