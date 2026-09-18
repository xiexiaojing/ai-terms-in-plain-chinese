#!/usr/bin/env bash
# 重新扫描文章 + 重建向量索引。新写了文章、或 feishu-claude-bot 抓了新快照之后跑一次。
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/ingest.py
.venv/bin/python scripts/build_index.py
.venv/bin/python scripts/build_graph.py
echo "索引已更新。服务在跑的话，重启一下才会生效：./daemon.sh restart"
