#!/usr/bin/env bash
# 在 www.ywcoding.cn 首页登记一条任务：建 task-*.md + 首页 index.html 插条目 + 推送 main + 确认线上生效
# 用法: tools/register_task.sh <slug> "<标题>" "<一句话描述>" [YYYY-MM-DD]
#   例: tools/register_task.sh dsh-simple "实现一个简单的 dsh" "做一个最小可用的 dsh 命令行工具" 2026-10-13
# 若 task-<slug>-<日期>.md 已存在则不覆盖，只补首页条目（可用于补登漏掉的任务）
set -euo pipefail
die() { echo "❌ $*" >&2; exit 1; }
[ $# -ge 3 ] || die "用法: $0 <slug> \"<标题>\" \"<一句话描述>\" [YYYY-MM-DD]"
SLUG=$1; TITLE=$2; DESC=$3; DATE=${4:-$(TZ=Asia/Shanghai date +%F)}
SITE=https://www.ywcoding.cn

cd "$(dirname "$0")/.."
# 1. 确认是真正的 clawbot_test 克隆（防止在空的假仓库里提交）
git remote get-url origin 2>/dev/null | grep -q "1540530942/clawbot_test" || die "$(pwd) 不是 clawbot_test 仓库"
[ -f index.html ] || die "$(pwd) 里没有 index.html，不是正确的克隆（spark 上请用 ~/clawbot_test）"
[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || git checkout -q main || die "切不到 main 分支"
git pull -q --rebase origin main || die "git pull 失败，先处理本地改动"

# 2. 任务详情 md（已存在就不动）
MD="task-$SLUG-$DATE.md"
if [ ! -f "$MD" ]; then
  cat > "$MD" <<MDEOF
# 任务记录：$TITLE

- **日期**：$DATE
- **状态**：今日任务
- **要求**：
- **任务内容**：$DESC
- **备注**：本条仅做任务登记，具体实现待后续展开
MDEOF
  echo "✅ 新建 $MD"
else
  echo "ℹ️  $MD 已存在，保留原内容"
fi

# 3. 首页条目（已有就跳过），插在笔记列表最顶上
python3 - "$MD" "$TITLE" "$DESC" "$DATE" <<'PY'
import sys, html, re
md, title, desc, date = sys.argv[1:]
p = "index.html"
t = open(p, encoding="utf-8").read()
if f'href="{md}"' in t:
    print("ℹ️  首页已有该条目"); sys.exit(0)
anchor = "<h2>笔记</h2>\n"
if anchor not in t:
    sys.exit("❌ index.html 里找不到 <h2>笔记</h2>，结构变了，停止")
e = html.escape
entry = (f'      <div class="entry">\n        <time>{date}</time>\n'
         f'        <h3>任务：{e(title)}</h3>\n'
         f'        <p>{e(desc)}。详见\n'
         f'        <a href="{md}" style="color:var(--accent2)">{md}</a>。</p>\n      </div>\n')
t = t.replace(anchor, anchor + entry, 1)
o, c = len(re.findall(r"<div", t)), len(re.findall(r"</div>", t))
if o != c or "<<<<<<<" in t:
    sys.exit(f"❌ 标签检查失败 div {o}/{c}，未写入")
open(p, "w", encoding="utf-8").write(t)
print(f"✅ 首页已插入条目（div {o}/{c} 配对）")
PY

# 4. 提交并推送到 main（网站只部署 main）
git add "$MD" index.html
if git diff --cached --quiet; then
  echo "ℹ️  没有新改动需要提交"
else
  git commit -q -m "task: 登记「$TITLE」($DATE)"
  git push -q origin main || { git pull -q --rebase origin main && git push -q origin main; } || die "push 失败"
  echo "✅ 已推送 $(git rev-parse --short HEAD) 到 main"
fi

# 5. 确认线上：tang 每 2 分钟同步一次，最多等 5 分钟。以首页出现条目为准，webhook 404 不用管
echo "⏳ 等待线上生效…"
for i in $(seq 1 30); do
  if curl -fsS --max-time 15 "$SITE/" 2>/dev/null | grep -q "href=\"$MD\"" \
     && [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$SITE/$MD")" = 200 ]; then
    echo "✅ 线上已生效：$SITE/ 首页可见，$SITE/$MD 可访问"; exit 0
  fi
  sleep 10
done
die "5 分钟内线上未出现该条目。推送已完成，请检查 tang 上的 /var/log/clawbot-deploy.log"
