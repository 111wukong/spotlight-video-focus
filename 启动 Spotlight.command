#!/bin/bash
# 双击这个文件就能启动 Spotlight 的网页界面。
set -e
cd "$(dirname "$0")"

PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "首次运行，正在准备 Python 环境…"
  if ! command -v uv >/dev/null 2>&1; then
    echo "没有找到 uv。请先安装：brew install uv"
    echo "（或者手动执行：python3.14 -m venv .venv && .venv/bin/pip install -e .）"
    read -r -p "按回车退出…" _
    exit 1
  fi
  uv venv --python 3.14 .venv
  echo "正在安装依赖（第一次会稍慢）…"
  uv pip install --python "$PY" -e .
fi

if [ ! -d models ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
  echo "还没有模型权重，先下载「均衡」档…"
  "$PY" -m spotlight.cli download --preset balanced
fi

exec "$PY" -m spotlight.cli serve "$@"
