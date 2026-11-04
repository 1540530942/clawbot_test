---
name: ywcoding-deploy
description: 把内容发布到 www.ywcoding.cn，或排查网页没更新。当要在 clawbot_test 仓库新增任务、笔记或页面并上线，确认合入后线上是否生效，或用户说"网页没更新"时使用。
---

# 发布到 www.ywcoding.cn

网站内容就是 `1540530942/clawbot_test` 的 `main` 分支。合入 `main` 就是公开发布，所以推送前确认内容可以公开，推送需要用户同意。

合入之后会自动发生：

1. GitHub Actions 的 `Sync site` 给首页补上新 `task-*.md` 的条目（`tools/site_sync.py sync`）。
2. tang 上的 cron 每 2 分钟同步一次 `main`。
3. 同一个 workflow 比对线上和仓库的文件，最多等 7 分钟，没更新就标红失败。

所以新增任务只需把 `task-<slug>-<YYYY-MM-DD>.md` 合入 `main`。标题取自 `# 任务记录：<标题>`，描述取自 `## 任务内容` 下的第一行。

## 发布之外要手动处理的情况

- 不想上首页的任务：把文件名写进 `.homepage-ignore`。
- 其他类型的新页面（非 `task-*.md`）：手动在 `index.html` 加入口，自动登记不处理它们。
- 论文精读：写成 `paperreading/NN-<slug>.html`，运行 `python3 tools/build_paperreading.py` 生成栏目首页，详见 `ywcoding-deploy.md`。
- 首页条目想改写：直接改 `index.html`。条目已存在时不会被覆盖。
- 删除任务：同时删掉 `index.html` 里的条目，自动登记不处理删除。

## 确认是否上线

本地跑，比对首页和指定提交之后改动的文件，通过时打印 `✅ 线上已生效`：

```bash
python3 tools/site_sync.py verify --since <合入前的提交>
```

只有这个命令通过，或 `Sync site` 运行成功，才能告诉用户已上线。推送成功不代表已上线。

## 排查"网页没更新"

按这个顺序查：

1. 改动是否在 `main` 上。`master` 不会部署。
2. 文件能打开但首页没有：看 `Sync site` 的运行记录，再看文件名是否在 `.homepage-ignore` 里、是否符合 `task-<slug>-<YYYY-MM-DD>.md`。
3. 文件也打不开，或 verify 超时：tang 上的 cron 同步出了问题，看 tang 上的 `/var/log/clawbot-deploy.log`。

Hermes 的操作说明和历史踩坑记录见 `ywcoding-deploy.md`，两者冲突时以它为准。
