@echo off
setlocal

if not exist node_modules (
  npm install
)

npm start
