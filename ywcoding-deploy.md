# 网页更新 / 部署指南（给 Hermes 用）

> 2026-10-14 建立。这是"怎么把内容更新到 www.ywcoding.cn"的**权威说明**，放在仓库里，不会被 curator 合并或归档。技能文件和这里冲突时，以这里为准。

## 一句话

**登记任务一律跑脚本，不要手动改文件**（在 spark 上）：

```bash
~/clawbot_test/tools/register_task.sh <slug> "<标题>" "<一句话描述>" [YYYY-MM-DD]
```

例：

```bash
~/clawbot_test/tools/register_task.sh dsh-simple "实现一个简单的 dsh" "做一个最小可用的 dsh 命令行工具"
```

脚本会自动完成：确认在正确的仓库 → `git pull` → 新建 `task-<slug>-<日期>.md` → 在首页 `index.html` 的笔记列表最顶上插入条目 → 检查 `<div>` 配对 → 推送到 `main` → **等待并确认线上首页已出现该条目**。最后一行打印 `✅ 线上已生效` 才算完成，否则就是失败，把输出原样告诉用户。

- 日期不写默认今天（北京时间）
- 如果 `task-*.md` 已经存在（比如之前漏登了首页），重跑同一条命令即可，只会补首页条目，不会覆盖原文件

## 部署原理（知道就行，不用手动操作）

- 网站内容 = GitHub 仓库 `1540530942/clawbot_test` 的 **`main` 分支**
- 托管服务器 tang 上的 cron 每 2 分钟下载 `main` 分支 tarball 并同步到网站目录，所以推送后 **最多约 2 分钟**生效
- **首页列表在 `index.html` 里**：每次推送到 `main`，GitHub Actions 的 `Sync site` 会给还没登记的 `task-*.md` 补上首页条目，再确认线上已更新，没更新时这次运行会失败。其他类型的页面仍要手动加入口
- 不想上首页的任务，把文件名写进 `.homepage-ignore`

## 常见坑（都发生过）

| 坑 | 正确做法 |
|----|----------|
| 手动合入 `task-*.md` 后首页没有条目 | 看 `Sync site` 的运行记录，或检查文件名是否在 `.homepage-ignore` 里、是否符合 `task-<slug>-<日期>.md` |
| 在 spark 的 `/home/archer/workspace/claw_bot/test/clawbot_test` 里提交 | 那是个空的假仓库（没有 index.html），**真正的克隆是 `~/clawbot_test`**，脚本会自动拒绝在假仓库里运行 |
| 推到了 `master` 分支 | 网站只部署 `main`，脚本只推 `main` |
| `POST /__deploy` webhook 返回 404，以为部署坏了 | webhook 早就不用了，**404 不代表失败**，以首页是否出现条目为准 |
| `git push` 报 `Permission denied (publickey)`，但 `ssh -T git@github.com` 正常 | GitHub 对同一把密钥偶尔会拒绝，隔 15 秒重试几次即可，不用改 SSH 配置 |
| 用 `execute_code` 抓网页检查 | 会卡在审批上；检查用 `curl -s https://www.ywcoding.cn/ \| grep "task-xxx.md"` 即可 |

## 手动检查某条是否上线

```bash
curl -s https://www.ywcoding.cn/ | grep -c 'href="task-<slug>-<日期>.md"'   # 1 = 首页已显示
curl -s -o /dev/null -w '%{http_code}\n' https://www.ywcoding.cn/task-<slug>-<日期>.md   # 200 = 文件已上线
```

## 记忆页面 memory.html

`memory.html` 展示 clawbot 的长期记忆（`~/.hermes/memories/` 下的 USER.md / MEMORY.md），由脚本生成，**不要手动改**。记忆有变化后在 spark 上跑：

```bash
~/clawbot_test/tools/publish_memory.sh
```

它会重新生成页面，记忆没变就什么也不做；有变化就提交、推送 `main`，并确认线上已生效。生成时会把看起来像密码/token 的内容替换成「[已隐藏]」，但写进记忆的内容会公开在网上，敏感信息本来就不该进记忆。

## 论文分享 paperreading/

每周一篇大模型经典论文精读，每期一个页面：`paperreading/NN-<slug>.html`，样式共用 `paperreading/style.css`。栏目首页 `paperreading/index.html` 由脚本生成，**不要手动改**。发新一期：

```bash
python3 tools/build_paperreading.py --next   # 看下一期是第几期
# 写好 paperreading/NN-<slug>.html，<head> 里带 paper:issue、short、year、title、summary 五个 meta
python3 tools/build_paperreading.py          # 检查文章并重新生成栏目首页
```

第 15 期起是图解版：初中生能读懂，有生活类比、逐步算例（`.calc`）、符号说明（`.syms`）和一个能动手玩的小组件（`.try`，内联脚本）。样式都在 `style.css` 里，写之前先读一篇现成的，比如 `31-ddpm.html`。

全部 50 期都是“全文精读版”（顶部 `<span>全文精读版</span>`），新一期照同样的标准写：

- 读完论文全文和附录；没读完的部分在“本期读法”和“来源”里写明。关于论文的说法要能在原文里找到，章节号要核对。
- 结构：溯源表 → 原理和公式（`.eq`、`.syms`、`.calc` 算例）→ 全部实验 → “读结果时要注意的地方”（论文的笔误和不一致）→ 亲手试 → 后来 → 局限 → 论文地图 → 一页总结。
- 亲手试是真跑的 NumPy 迷你实验，脚本放 `paperreading/code/NN_mini_x.py`。页面组件用实验里训练好的权重，在浏览器里跑，和 Python 输出要对得上。
- 结果下面放 `.why` 框“这组结果怎么读（诚实版）”：和论文一致、不一致的都写；失败的运行和重跑的原因也写，数字留在小表里；没验证的解释标“猜测”。不编数字。


然后提交并推送 `main`。期数看仓库里已发布到第几期，不按日期算。页面里不要引用 Google Fonts 等外部资源，也不要链接 claude.ai 上的私有页面。
