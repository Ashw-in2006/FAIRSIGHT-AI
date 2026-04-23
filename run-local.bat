@echo off
setlocal

start "FairSight Backend" cmd /k "cd /d ""%~dp0backend"" && call start.bat"
start "FairSight Frontend" cmd /k "cd /d ""%~dp0frontend"" && call start.bat"
