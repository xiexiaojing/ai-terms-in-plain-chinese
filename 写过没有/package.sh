#!/usr/bin/env bash
# 打一份能直接发给学员的包。
# 唯一要盯死的是 .env——里面有 API key，绝不能跟着走。
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-$HOME/Desktop/gzh-rag-demo.tar.gz}"
STAGE="$(mktemp -d)/gzh-rag"
mkdir -p "$STAGE"

# 白名单：只有列进来的才进包，比黑名单安全——以后新增文件不会被默默带出去
for item in scripts web data docs README.md requirements.txt .env.example \
            run.sh daemon.sh refresh.sh openspec 我的文件; do
  [ -e "$item" ] && cp -R "$item" "$STAGE/"
done

# 兜底自查：万一将来有人往白名单里加了带密钥的东西
find "$STAGE" -name ".env" -o -name "*.key" -o -name "*.pem" | grep . && {
  echo "✗ 包里有密钥文件，已中止"; exit 1; }
if grep -rlE "sk-[a-zA-Z0-9_-]{20,}" "$STAGE" 2>/dev/null | grep -v ".env.example"; then
  echo "✗ 包里有疑似 API key，已中止"; exit 1
fi

rm -rf "$STAGE"/**/__pycache__ "$STAGE"/scripts/__pycache__ 2>/dev/null || true
tar -czf "$OUT" -C "$(dirname "$STAGE")" gzh-rag
rm -rf "$(dirname "$STAGE")"
echo "✓ $OUT  ($(du -h "$OUT" | cut -f1))"
echo "  学员解开后：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && ./daemon.sh start"
