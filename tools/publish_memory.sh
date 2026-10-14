#!/bin/bash
# Regenerate memory.html from clawbot's live memory and push it if it changed.
# Safe to run any time (no-op when nothing changed). Run on spark:
#   ~/clawbot_test/tools/publish_memory.sh
set -euo pipefail
REPO="$HOME/clawbot_test"
cd "$REPO"
[ -f index.html ] || { echo "❌ $REPO 不是网站仓库"; exit 1; }

git pull -q --rebase origin main
python3 tools/build_memory_page.py memory.html

# Ignore the timestamp line when deciding whether anything really changed.
if git diff --quiet -I '快照时间' -- memory.html && git ls-files --error-unmatch memory.html >/dev/null 2>&1; then
  git checkout -- memory.html
  echo "记忆没有变化，不需要更新"
  exit 0
fi

git add memory.html
git commit -q -m "memory: 同步 clawbot 记忆快照 ($(TZ=Asia/Shanghai date +%F))"
git push -q origin main

stamp=$(grep -o '快照时间 [0-9-]* [0-9:]*' memory.html)
for _ in $(seq 1 18); do
  if curl -fsS https://www.ywcoding.cn/memory.html 2>/dev/null | grep -q "$stamp"; then
    echo "✅ 线上已生效：https://www.ywcoding.cn/memory.html"
    exit 0
  fi
  sleep 10
done
echo "⚠️ 已推送，但 3 分钟内线上未看到新版本，请手动检查"
exit 1
