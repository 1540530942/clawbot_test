#!/usr/bin/env python3
"""Render clawbot's (Hermes Agent) persistent memory into memory.html.

Reads ~/.hermes/memories/{MEMORY,USER}.md (entries split by "\\n§\\n") and the
char limits from ~/.hermes/config.yaml, then writes a static page that matches
index.html's look. Run from the repo root:  python3 tools/build_memory_page.py
Re-run whenever the memory changes; tools/publish_memory.sh does that + push.
"""
import datetime
import html
import os
import re
import sys

HERMES = os.path.expanduser("~/.hermes")
DELIM = "\n§\n"
OUT = sys.argv[1] if len(sys.argv) > 1 else "memory.html"

# Never publish anything that looks like a credential, even if it slips into memory.
SECRET_RE = re.compile(
    r"(?i)((?:password|passwd|密码|token|api[_-]?key|secret|psk)\s*[:=：]\s*)\S+")


def read_entries(name):
    path = os.path.join(HERMES, "memories", name)
    try:
        raw = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return [], 0
    entries = [e.strip() for e in raw.split(DELIM) if e.strip()]
    return entries, len(DELIM.join(entries))


def read_limits():
    limits = {"memory_char_limit": 2200, "user_char_limit": 1375}
    try:
        text = open(os.path.join(HERMES, "config.yaml"), encoding="utf-8").read()
        for key in limits:
            m = re.search(rf"^\s*{key}:\s*(\d+)", text, re.M)
            if m:
                limits[key] = int(m.group(1))
    except FileNotFoundError:
        pass
    return limits


def classify(entry):
    head = entry[:12]
    if "硬约束" in head:
        return "hard", "硬约束"
    if "约束" in head:
        return "rule", "约束"
    if "偏好" in head:
        return "pref", "偏好"
    return "know", "做法"


def render_entry(entry, kind=None):
    text = SECRET_RE.sub(r"\1[已隐藏]", entry)
    text = text.lstrip("⚠️ ").strip()
    date = re.search(r"20\d\d-\d\d-\d\d", text)
    body = html.escape(text)
    body = re.sub(r"`([^`]+)`", r"<code>\1</code>", body)
    body = body.replace("\n", "<br>")
    if kind is None:
        kind, label = classify(entry)
    else:
        label = {"profile": "画像"}.get(kind, "")
    meta = f'<time>{date.group(0)}</time>' if date else ""
    return (f'<li class="mem {kind}"><div class="mem-meta">'
            f'<span class="tag">{label}</span>{meta}</div>'
            f'<p>{body}</p></li>')


def meter(used, limit):
    pct = min(100, round(used * 100 / limit)) if limit else 0
    return (f'<div class="meter"><div class="meter-bar"><span style="width:{pct}%"></span></div>'
            f'<span class="meter-num">{used:,} / {limit:,} 字符 · {pct}%</span></div>')


def main():
    mem, mem_used = read_entries("MEMORY.md")
    usr, usr_used = read_entries("USER.md")
    lim = read_limits()
    order = {"hard": 0, "rule": 1, "pref": 2, "know": 3}
    mem_sorted = sorted(mem, key=lambda e: order[classify(e)[0]])
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    stamp = now.strftime("%Y-%m-%d %H:%M")

    mem_items = "\n".join(render_entry(e) for e in mem_sorted)
    usr_items = "\n".join(
        render_entry(e, None if "偏好" in e[:12] else "profile") for e in usr)

    page = TEMPLATE.format(
        stamp=stamp,
        usr_meter=meter(usr_used, lim["user_char_limit"]),
        mem_meter=meter(mem_used, lim["memory_char_limit"]),
        usr_items=usr_items, mem_items=mem_items,
        n_usr=len(usr), n_mem=len(mem))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {OUT}: USER {len(usr)} entries {usr_used}c, MEMORY {len(mem)} entries {mem_used}c")


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>clawbot 的记忆</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #0a0a0f; --surface: #111118; --border: #1e1e2e;
      --accent: #4fd1c5; --accent2: #7c6af7; --warn: #f0a35e;
      --text: #e2e2f0; --muted: #6b6b8a;
    }}
    body {{ background: var(--bg); color: var(--text);
      font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif; line-height: 1.7; }}
    nav {{ position: sticky; top: 0; z-index: 100; display: flex; align-items: center;
      justify-content: space-between; padding: 18px 48px; background: rgba(10,10,15,0.85);
      backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); }}
    .logo {{ font-size: 1.2rem; font-weight: 700; letter-spacing: -0.5px; }}
    .logo span {{ color: var(--accent); }}
    nav a {{ color: var(--muted); text-decoration: none; font-size: 0.85rem; }}
    nav a:hover, nav a:focus-visible {{ color: var(--text); }}
    main {{ max-width: 860px; margin: 0 auto; padding: 56px 24px 100px; }}
    header.page {{ margin-bottom: 40px; }}
    h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 10px; text-wrap: balance; }}
    header.page p {{ color: var(--muted); font-size: 0.95rem; max-width: 640px; }}
    header.page .stamp {{ margin-top: 8px; font-size: 0.8rem; font-variant-numeric: tabular-nums; }}
    section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
      padding: 28px 32px; margin-bottom: 24px; }}
    section h2 {{ font-size: 1.2rem; display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
    section h2 small {{ color: var(--muted); font-weight: 400; font-size: 0.8rem; }}
    section > .lede {{ color: var(--muted); font-size: 0.85rem; margin: 4px 0 14px; }}
    .meter {{ display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }}
    .meter-bar {{ flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }}
    .meter-bar span {{ display: block; height: 100%; background: var(--accent); }}
    .meter-num {{ color: var(--muted); font-size: 0.78rem; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    ul.mems {{ list-style: none; display: grid; gap: 14px; }}
    .mem {{ border-left: 2px solid var(--accent2); padding: 2px 0 2px 16px; }}
    .mem.hard {{ border-left-color: var(--warn); }}
    .mem.rule {{ border-left-color: var(--warn); }}
    .mem.profile, .mem.pref {{ border-left-color: var(--accent); }}
    .mem-meta {{ display: flex; gap: 10px; align-items: center; margin-bottom: 2px; }}
    .tag {{ font-size: 0.7rem; letter-spacing: 0.06em; padding: 1px 8px; border-radius: 999px;
      border: 1px solid var(--border); color: var(--muted); }}
    .hard .tag, .rule .tag {{ color: var(--warn); border-color: rgba(240,163,94,0.4); }}
    .profile .tag, .pref .tag {{ color: var(--accent); border-color: rgba(79,209,197,0.4); }}
    .mem time {{ color: var(--muted); font-size: 0.75rem; font-variant-numeric: tabular-nums; }}
    .mem p {{ color: #c4c4d8; font-size: 0.92rem; overflow-wrap: anywhere; }}
    code {{ font-family: ui-monospace, 'SFMono-Regular', Menlo, monospace; font-size: 0.85em;
      background: #181824; padding: 1px 5px; border-radius: 4px; color: var(--text); }}
    .how {{ color: #c4c4d8; font-size: 0.9rem; }}
    .how li {{ margin-left: 20px; margin-top: 4px; }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; padding: 32px;
      border-top: 1px solid var(--border); }}
    footer a {{ color: var(--accent); text-decoration: none; }}
    @media (max-width: 600px) {{
      nav {{ padding: 14px 16px; }} main {{ padding: 36px 16px 80px; }}
      section {{ padding: 20px 18px; }} h1 {{ font-size: 1.6rem; }}
    }}
  </style>
</head>
<body>
  <nav>
    <div class="logo">clawbot<span>_test</span></div>
    <a href="./">← 返回笔记</a>
  </nav>
  <main>
    <header class="page">
      <h1>clawbot 记住了什么</h1>
      <p>这是微信 bot（Hermes Agent）的长期记忆原文。每次开新会话时，它会把这两份记忆整段放进提示词，
      所以这里写的就是它“开口前已经知道的事”。</p>
      <p class="stamp">快照时间 {stamp}（北京时间）· 由 <code>tools/build_memory_page.py</code> 从 spark 上的记忆文件生成</p>
    </header>

    <section>
      <h2>关于你 <small>USER.md · {n_usr} 条</small></h2>
      <p class="lede">它对使用者的画像和偏好。</p>
      {usr_meter}
      <ul class="mems">
{usr_items}
      </ul>
    </section>

    <section>
      <h2>规则与做法 <small>MEMORY.md · {n_mem} 条</small></h2>
      <p class="lede">它必须遵守的约束（橙色），以及踩坑后总结出的工作方式。</p>
      {mem_meter}
      <ul class="mems">
{mem_items}
      </ul>
    </section>

    <section class="how">
      <h2>这份记忆怎么运作</h2>
      <ul>
        <li>只在会话开始时读取一次：改了记忆，正在进行的会话要等下次轮换才会生效。</li>
        <li>有字数上限（见上方进度条），写满后它就记不进新东西，需要清理旧条目。</li>
        <li>一次性的问答、过期的快照不该进记忆——2026-10-15 已清理过一轮。</li>
      </ul>
    </section>
  </main>
  <footer>
    由 clawbot_test 自主维护 · 托管在 <a href="https://www.ywcoding.cn/">ywcoding.cn</a>
  </footer>
</body>
</html>
"""

if __name__ == "__main__":
    main()
