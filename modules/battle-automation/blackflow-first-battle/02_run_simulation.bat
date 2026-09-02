@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -m blackflow.cli --config configs\strategy_first_battle.example.json --simulate --episodes 80
pause

