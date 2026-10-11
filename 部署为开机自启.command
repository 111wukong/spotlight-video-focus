#!/bin/bash
# 把 Spotlight 注册成 macOS 登录自启服务。
#
# 为什么要你自己双击运行：从 WorkBuddy 内部发起的进程执行 launchctl 会被系统拒绝
# （报 "5: Input/output error"），只有你自己的终端才有权限注册 LaunchAgent。
#
# 双击本文件即可；想取消自启就双击「停止并卸载服务.command」。
set -e
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
LABEL="ai.spotlight.server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PORT="${SPOTLIGHT_PORT:-8760}"

echo "Spotlight 部署为开机自启"
echo "────────────────────────────────────────────"
echo "  项目目录  $PROJECT_DIR"
echo "  解释器    $PYTHON"
echo "  端口      $PORT"
echo "  服务标识  $LABEL"
echo

if [ ! -x "$PYTHON" ]; then
  echo "✗ 找不到 Python 环境：$PYTHON"
  echo "  请先双击「启动 Spotlight.command」跑一次，它会自动把环境建好。"
  read -r -p "按回车退出…" _
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/work/logs"

# 先卸掉旧的，避免重复注册报错
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl unload "$PLIST" 2>/dev/null || true

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON</string>
        <string>-m</string>
        <string>spotlight.cli</string>
        <string>serve</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>$PORT</string>
        <string>--no-browser</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/work/logs/server.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/work/logs/server.err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
PLIST_EOF

echo "  已写入 $PLIST"
echo

if launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl load -w "$PLIST" 2>/dev/null; then
  sleep 3
  if curl -s -o /dev/null --noproxy '*' "http://127.0.0.1:$PORT/api/status" 2>/dev/null; then
    echo "✓ 部署成功，服务已经在跑，并且以后每次登录都会自动启动。"
    echo
    echo "  打开：http://127.0.0.1:$PORT/"
    echo "  日志：$PROJECT_DIR/work/logs/server.log"
    echo "  卸载：双击「停止并卸载服务.command」"
    open "http://127.0.0.1:$PORT/" 2>/dev/null || true
  else
    echo "⚠ 服务已注册但暂时没响应，稍等几秒再打开 http://127.0.0.1:$PORT/"
    echo "  若一直不行，看日志：$PROJECT_DIR/work/logs/server.err.log"
  fi
else
  echo "✗ 注册失败。可以手动执行："
  echo "    launchctl bootstrap gui/\$(id -u) \"$PLIST\""
fi
echo
read -r -p "按回车关闭窗口…" _
