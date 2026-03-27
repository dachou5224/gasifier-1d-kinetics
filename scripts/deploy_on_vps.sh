#!/usr/bin/env bash
# 用法:
#   ./scripts/deploy_on_vps.sh [REPO_PATH]
# 示例:
#   ./scripts/deploy_on_vps.sh /root/gasifier-1d-kinetic

set -euo pipefail

REPO_PATH="${1:-/root/gasifier-1d-kinetic}"

echo "Deploy path: $REPO_PATH"
if [ ! -d "$REPO_PATH/.git" ]; then
  echo "ERROR: $REPO_PATH 不是 Git 仓库目录" >&2
  exit 1
fi

cd "$REPO_PATH"
git fetch origin
git checkout main
git pull --ff-only origin main

echo "Done. Current commit: $(git rev-parse --short HEAD)"
