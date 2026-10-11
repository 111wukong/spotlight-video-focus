#!/bin/bash
# 停止 Spotlight 并取消开机自启。
#
# 会做两件事：卸掉 LaunchAgent（如果装过），以及杀掉手动启动的实例。
set -e
cd "$(dirname "$0")"
LABEL="ai.spotlight.server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PORT="${SPOTLIGHT_PORT:-8760}"

echo "停止 Spotlight"
echo "────────────────────────────────────────────"

if [ -f "$PLIST" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null \
    || launchctl unload "$PLIST" 2>/dev/null \
    || true
  rm -f "$PLIST"
  echo "  ✓ 已取消开机自启并移除 $PLIST"
else
  echo "  · 没有装过开机自启"
fi

# 再兜底杀掉可能手动启动的实例
PIDS="$(pgrep -f "spotlight.cli serve" || true)"
if [ -n "$PIDS" ]; then
  # shellcheck disable=SC2086
  kill $PIDS 2>/dev/null || true
  sleep 1
  # shellcheck disable=SC2086
  kill -9 $PIDS 2>/dev/null || true
  echo "  ✓ 已停止手动启动的实例（PID $PIDS）"
fi

sleep 1
if curl -s -o /dev/null --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/api/status" 2>/dev/null; then
  echo
  echo "⚠ 端口 $PORT 仍然有响应，可能还有别的实例在跑："
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tail -2
else
  echo
  echo "✓ 服务已完全停止。"
fi
echo
read -r -p "按回车关闭窗口…" _
