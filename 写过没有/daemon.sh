#!/usr/bin/env bash
# 后台常驻：./daemon.sh start|stop|restart|status  日志在 logs/server.log
cd "$(dirname "$0")"
PORT=${PORT:-7788}
PIDFILE=.server.pid
mkdir -p logs

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; }

case "${1:-start}" in
  start)
    if running; then echo "已经在跑（pid $(cat $PIDFILE)）：http://127.0.0.1:$PORT"; exit 0; fi
    # .env 里放 MINIMAX_API_KEY，不进 git
    [ -f .env ] && { set -a; . ./.env; set +a; }
    HF_HUB_OFFLINE=1 nohup .venv/bin/python scripts/server.py "$PORT" >> logs/server.log 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 60); do
      curl -sf -m 1 "http://127.0.0.1:$PORT/api/stats" >/dev/null && break
      sleep 1
    done
    echo "已启动：http://127.0.0.1:$PORT  (pid $(cat $PIDFILE))"
    ;;
  stop)
    running && { kill "$(cat $PIDFILE)"; echo "已停止"; } || echo "没在跑"
    rm -f "$PIDFILE"
    ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  status)
    running && echo "在跑（pid $(cat $PIDFILE)）：http://127.0.0.1:$PORT" || echo "没在跑"
    ;;
esac
