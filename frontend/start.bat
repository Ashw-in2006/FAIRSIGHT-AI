@echo off
setlocal

if not exist node_modules (
  call reset-node-modules.bat
)

npm start
