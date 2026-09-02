@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -m blackflow.cli --config configs\strategy_first_battle.json --episodes 1
pause

