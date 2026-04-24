@echo off
setlocal

if exist node_modules (
  rmdir /s /q node_modules
)

if exist package-lock.json (
  del /f /q package-lock.json
)

npm install
