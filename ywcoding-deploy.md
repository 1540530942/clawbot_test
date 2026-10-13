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
- **首页列表是手写在 `index.html` 里的**：只加 `task-*.md` 而不改 `index.html`，文件虽然能直接打开，但首页看不到——这是过去多次"网页没更新"的真正原因

## 常见坑（都发生过）

| 坑 | 正确做法 |
|----|----------|
| 只提交了 `task-*.md`，没改 `index.html` | 用脚本，它会同时改两个文件 |
| 在 spark 的 `/home/archer/workspace/claw_bot/test/clawbot_test` 里提交 | 那是个空的假仓库（没有 index.html），**真正的克隆是 `~/clawbot_test`**，脚本会自动拒绝在假仓库里运行 |
| 推到了 `master` 分支 | 网站只部署 `main`，脚本只推 `main` |
| `POST /__deploy` webhook 返回 404，以为部署坏了 | webhook 早就不用了，**404 不代表失败**，以首页是否出现条目为准 |
| 用 `execute_code` 抓网页检查 | 会卡在审批上；检查用 `curl -s https://www.ywcoding.cn/ \| grep "task-xxx.md"` 即可 |

## 手动检查某条是否上线

```bash
curl -s https://www.ywcoding.cn/ | grep -c 'href="task-<slug>-<日期>.md"'   # 1 = 首页已显示
curl -s -o /dev/null -w '%{http_code}\n' https://www.ywcoding.cn/task-<slug>-<日期>.md   # 200 = 文件已上线
```
