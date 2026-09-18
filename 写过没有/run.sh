#!/usr/bin/env bash
# 前台启动检索服务，Ctrl-C 退出。默认端口 7788，被占用会自动顺延。
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
exec .venv/bin/python scripts/server.py "${1:-7788}"
