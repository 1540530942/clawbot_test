# Hermes ⇄ kcc 专属聊天通道

> 2026-08-26 建立，2026-09-27 改为按天轮换会话（不再用固定永久会话 ID），2026-09-29 改为统一用封装脚本 kcc_chat.sh。本文件为通道说明存档，是 Hermes 判断"怎么联系 kcc"时的权威参考——**改动机制时改这份文件，不要只改 `~/.hermes/skills/` 里的技能文件**。

## 背景

- **Hermes**：运行在 spark 工作站上的 Agent，经微信与用户交互，可 ssh 到 korea。
- **kcc**：korea 服务器（VM-0-5-ubuntu）上的 Claude Code CLI（`/usr/local/bin/claude`，v2.1.177，Claude Pro 账号），用户起的代号。

两者不经任何外部聊天平台，直接通过 korea 上的 Claude Code 会话通信：

```
用户(微信) ⇄ Hermes(spark) --ssh korea--> claude -p [--resume <今天的会话ID> | --session-id <新UUID>] ⇄ kcc 的上下文记忆
```

**⚠️ 已废弃**：旧方案用一个永久固定的会话 ID(`9db3280e-7927-4c54-bb74-bcee66d1064b`)，指望一直沿用下去——实测它被晾了 23 天没人碰，而且没法确认"还有效吗"。**现在改成按天轮换**，见下。

## 通道规格（2026-09-27 起，按天轮换）

- **指针文件**（记录"今天用的是哪个会话"，在 spark 本机）：
  `/home/archer/workspace/chat_bots/history/weixin_hermes_bot/current_session.json`
  格式：`{"date": "YYYY-MM-DD", "session_id": "<uuid>", "updated": "<ISO时间戳>"}`
- **对话记录**（每天一个文件夹，人类可读）：
  `/home/archer/workspace/chat_bots/history/weixin_hermes_bot/<YYYY-MM-DD>/conversation.md`
- **工作目录**（korea 上，不变）：`/home/ubuntu/workspace/claw_bot/test/`

## 发送方式（2026-09-29 起：只跑一条命令）

**轻聊天/问答一律用封装脚本，不要自己拼 `claude -p` 命令**（在 spark 上执行）：

```bash
/home/archer/workspace/chat_bots/bin/kcc_chat.sh "<发给 kcc 的消息>"
```

脚本已自动处理全部细节：读指针文件判断今天有没有会话 → 有则 `--resume`、没有则 `uuidgen` + `--session-id` 新建 → `--max-turns 10`（允许 kcc 调工具，比如 `date`）→ 覆写指针文件 → 把这一轮追加进当天 `conversation.md`。标准输出就是 kcc 的回复，退出码非 0 表示失败。

- 用户说"问 kcc / 让 kcc 说 / 发给 kcc"时，就是调用这个脚本，**不要自己 ssh 到 korea 查答案代替 kcc 回答**
- 重活（多步任务）：另起独立一次性会话 + nohup 后台跑，跑完把结果带回——不走这个脚本，也不用碰指针文件
- 已实测（2026-09-29）：新建分支 + 续聊分支都跑通，kcc 能记住同一天内上一轮内容

## 边界

- kcc 只在 `test/` 目录内操作（`CLAUDE.md` 软约束）
- 通道不承载敏感凭据；密钥/token 一律不进对话内容
- 完整机制细节另存了一份技能文档：`~/.hermes/skills/devops/kcc-chat-channel/SKILL.md`（但这份 `kcc-channel.md` 才是主参考，两边要保持一致，改一处记得回来改另一处）
