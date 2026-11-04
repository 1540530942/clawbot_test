#!/usr/bin/env python3
"""合入 main 后让 www.ywcoding.cn 跟上仓库。由 .github/workflows/sync-site.yml 调用，也可手动跑。

  site_sync.py sync                 给首页补上还没登记的 task-*.md 条目，已有的不动
  site_sync.py verify [--since SHA] 等线上和仓库一致：比对 index.html 及 SHA 之后改动的文件

有意不上首页的任务，把文件名写进 .homepage-ignore。
"""
import argparse
import hashlib
import html
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = ROOT / "index.html"
IGNORE = ROOT / ".homepage-ignore"
SITE = "https://www.ywcoding.cn"
ANCHOR = "<h2>笔记</h2>\n"
TASK_NAME = re.compile(r"^task-(?P<slug>.+)-(?P<date>\d{4}-\d{2}-\d{2})(?:-v\d+)?\.md$")
DESC_MAX = 80


def clean(text):
    return re.sub(r"[*`]", "", text).strip().rstrip("。")


def parse_task(path, slug):
    """取标题和一句话描述。部分旧任务没有 # 标题，或把任务内容写在列表项里。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    title = next((l[2:] for l in lines if l.startswith("# ")), slug)
    title = clean(title.removeprefix("任务记录：")) or slug

    desc = ""
    for i, line in enumerate(lines):
        inline = re.match(r"-\s*\*\*任务内容\*\*[：:]\s*(.+)", line)
        if inline:
            desc = inline.group(1)
            break
        if line.strip() == "## 任务内容":
            desc = next((l for l in lines[i + 1:] if l.strip() and not l.startswith("#")), "")
            break
    desc = clean(desc) or title
    if len(desc) > DESC_MAX:
        desc = desc[:DESC_MAX] + "…"
    return title, desc


def render_entry(name, date, title, desc):
    e = html.escape
    return (f'      <div class="entry">\n        <time>{date}</time>\n'
            f'        <h3>任务：{e(title)}</h3>\n'
            f'        <p>{e(desc)}。详见\n'
            f'        <a href="{name}" style="color:var(--accent2)">{name}</a>。</p>\n      </div>\n')


def cmd_sync(_args):
    text = INDEX.read_text(encoding="utf-8")
    if ANCHOR not in text:
        sys.exit("❌ index.html 里找不到 <h2>笔记</h2>，结构变了，停止")

    ignored = set()
    if IGNORE.exists():
        for line in IGNORE.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                ignored.add(line)

    pending = []
    for path in ROOT.glob("task-*.md"):
        name = path.name
        if name in ignored or f'href="{name}"' in text:
            continue
        match = TASK_NAME.match(name)
        if not match:
            print(f"⚠️  跳过 {name}：文件名不是 task-<slug>-<YYYY-MM-DD>.md")
            continue
        pending.append((match["date"], name, match["slug"], path))

    # 逐条插到列表最顶上，所以按日期升序处理，最新的排在最前
    for date, name, slug, path in sorted(pending):
        title, desc = parse_task(path, slug)
        text = text.replace(ANCHOR, ANCHOR + render_entry(name, date, title, desc), 1)
        print(f"✅ 首页新增 {name}：{title}")

    if not pending:
        print("ℹ️  首页无需更新")
        return
    opened, closed = len(re.findall(r"<div", text)), len(re.findall(r"</div>", text))
    if opened != closed or "<<<<<<<" in text:
        sys.exit(f"❌ 标签检查失败 div {opened}/{closed}，未写入")
    INDEX.write_text(text, encoding="utf-8")


def changed_since(sha):
    """sha 之后新增或修改的文件。sha 不可用（如强推、手动触发）时返回空，只比对首页。"""
    if not sha or not sha.strip("0"):
        return []
    result = subprocess.run(
        ["git", "diff", "-z", "--name-only", "--diff-filter=AM", sha, "HEAD"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        print(f"⚠️  无法对比 {sha}，只检查首页")
        return []
    return [f for f in result.stdout.split("\0") if f]


def is_published(rel):
    # 点开头的路径（.github、.claude 等）是仓库配置，不算网站内容
    return not any(part.startswith(".") for part in pathlib.PurePosixPath(rel).parts)


def live_digest(site, rel):
    url = f"{site}/{urllib.parse.quote(rel)}"
    req = urllib.request.Request(url, headers={"User-Agent": "clawbot-site-sync"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return hashlib.sha256(resp.read()).hexdigest()
    except (urllib.error.URLError, TimeoutError) as err:
        return f"error: {err}"


def cmd_verify(args):
    files = {"index.html", *filter(is_published, changed_since(args.since))}
    files = {f for f in files if (ROOT / f).is_file()}
    expected = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}

    print(f"⏳ 等待线上生效（最多 {args.timeout} 秒），检查 {len(files)} 个文件")
    deadline = time.monotonic() + args.timeout
    while True:
        pending = sorted(f for f, digest in expected.items() if live_digest(args.site, f) != digest)
        if not pending:
            print(f"✅ 线上已生效：{args.site}/")
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(args.interval)
    sys.exit(f"❌ {args.timeout} 秒内线上仍未更新：{', '.join(pending)}。"
             "请检查 tang 上的 /var/log/clawbot-deploy.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sync").set_defaults(func=cmd_sync)
    verify = commands.add_parser("verify")
    verify.add_argument("--since", help="同时比对这个提交之后改动的文件")
    verify.add_argument("--site", default=SITE)
    verify.add_argument("--timeout", type=int, default=420)
    verify.add_argument("--interval", type=int, default=15)
    verify.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
