@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -m blackflow.calibrate screenshot --config configs\strategy_first_battle.json
pause

