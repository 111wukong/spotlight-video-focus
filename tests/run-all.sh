#!/bin/bash
# 跑一遍全部检查。任何一项失败就以非零退出。
#
# 需要浏览器 / 需要服务的项目在跑不了时会明确打印 [SKIP]，
# 不会假装通过，也不会让整套检查变红。
cd "$(dirname "$0")/.." || exit 1

NODE="${NODE:-$(command -v node)}"
PY=".venv/bin/python"
PORT="${SPOTLIGHT_TEST_PORT:-8791}"
rc=0

echo "════ 1/5 环境自检 ════"
$PY -m spotlight.cli doctor || rc=1
echo

echo "════ 2/5 核心行为回归 ════"
$PY tests/test-core.py || rc=1
echo

echo "════ 3/5 前端接线检查（静态，秒级）════"
if [ -n "$NODE" ]; then
  "$NODE" tests/check-wiring.js || rc=1
else
  echo "  · 没找到 node，跳过"
fi
echo

echo "════ 4/5 后端接口全链路 ════"
# 自己起一个服务、跑完关掉，不需要先手动启动
$PY -m spotlight.cli serve --port "$PORT" --no-browser >/tmp/spotlight-test-server.log 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 60); do
  if curl -s -o /dev/null --noproxy '*' "http://127.0.0.1:$PORT/api/status" 2>/dev/null; then break; fi
  sleep 0.5
done
$PY tests/api_smoke.py --base "http://127.0.0.1:$PORT" || rc=1
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo

echo "════ 5/5 浏览器冒烟（本环境会明确跳过）════"
if [ -n "$NODE" ]; then
  "$NODE" tests/smoke-ui.js || rc=1
else
  echo "  · 没找到 node，跳过"
fi
echo

if [ $rc -eq 0 ]; then
  echo "✅ 全部检查通过"
else
  echo "❌ 有检查未通过"
fi
exit $rc
