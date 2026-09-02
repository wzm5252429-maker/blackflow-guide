@echo off
chcp 65001 >nul
cd /d "%~dp0"
set /p EPISODES=How many battles should be attempted? 
py -m blackflow.cli --config configs\strategy_first_battle.json --episodes %EPISODES%
pause

