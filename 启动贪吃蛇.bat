@echo off
chcp 65001 >nul
cd /d %~dp0
"C:\Users\yu320\AppData\Local\Doubao\User Data\sandbox_runtime\bases\772984f6fd3a8ae0ae7d2422cd7c2bfa\python\python.exe" launcher.py
if errorlevel 1 pause