#!/usr/bin/env python3
"""根据 paperreading/NN-<slug>.html 里的 meta 标签生成论文栏目首页 paperreading/index.html。

  build_paperreading.py          检查所有文章并重新生成首页
  build_paperreading.py --next   只打印下一期的期数（已发布的最大期数 + 1）

文章 <head> 里要有 paper:issue、paper:short、paper:year、paper:title、paper:summary。
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "paperreading"
ARTICLE_NAME = re.compile(r"^(\d{2,})-[a-z0-9-]+\.html$")
FIELDS = ("issue", "short", "year", "title", "summary")

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>论文分享 · 每周经典论文</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="wrap"><article>
<header class="masthead">
  <div class="series"><a href="../">← www.ywcoding.cn</a><span>论文分享</span></div>
  <h1>每周经典论文<small>把一篇大模型经典论文的动机、设计和取舍讲透</small></h1>
  <p class="lede">从 Transformer 开始，沿着大模型的主线一篇一篇读。每期都是单独的页面，配图解、关键数字和来源，读者设定是做机器人和端侧部署的工程师。</p>
</header>
<ol class="plist" reversed>
{items}
</ol>
</article></div>
</body>
</html>
"""
ITEM = """  <li><a href="{name}"><span class="no">第 {issue} 期</span><span class="ttl">{short}<small>{year}</small></span><span class="orig">{title}</span><span class="sum">{summary}</span></a></li>"""


def read_article(path):
    text = path.read_text(encoding="utf-8")
    meta = {}
    for field in FIELDS:
        found = re.search(rf'<meta name="paper:{field}" content="([^"]*)">', text)
        if not found or not found.group(1).strip():
            sys.exit(f"❌ {path.name} 缺少 paper:{field}")
        meta[field] = found.group(1)
    match = ARTICLE_NAME.match(path.name)
    if int(match.group(1)) != int(meta["issue"]):
        sys.exit(f"❌ {path.name} 的文件名和 paper:issue={meta['issue']} 不一致")
    opened, closed = len(re.findall(r"<div\b", text)), len(re.findall(r"</div>", text))
    if opened != closed:
        sys.exit(f"❌ {path.name} 的 div 不配对：{opened}/{closed}")
    meta["name"] = path.name
    return meta


def main():
    articles = [read_article(p) for p in sorted(ROOT.glob("*.html")) if ARTICLE_NAME.match(p.name)]
    issues = [int(a["issue"]) for a in articles]
    if len(set(issues)) != len(issues):
        sys.exit(f"❌ 期数有重复：{sorted(issues)}")
    if sys.argv[1:] == ["--next"]:
        print(max(issues, default=0) + 1)
        return
    articles.sort(key=lambda a: int(a["issue"]), reverse=True)
    # meta 里的内容本来就是转义过的 HTML 属性值，这里原样放进页面
    items = "\n".join(ITEM.format(**a) for a in articles)
    (ROOT / "index.html").write_text(PAGE.format(items=items), encoding="utf-8")
    print(f"✅ 已生成 paperreading/index.html，共 {len(articles)} 期，下一期是第 {max(issues, default=0) + 1} 期")


if __name__ == "__main__":
    main()
