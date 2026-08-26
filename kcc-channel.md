# Hermes ⇄ kcc 专属聊天通道

> 2026-08-26 建立，本文件为通道说明存档。

## 背景

- **Hermes**：运行在 spark 工作站上的 Agent，经微信与用户交互，可 ssh 到 korea。
- **kcc**：korea 服务器（VM-0-5-ubuntu）上的 Claude Code CLI（`/usr/local/bin/claude`，v2.1.177，Claude Pro 账号），用户起的代号。

两者不经任何外部聊天平台，直接通过 korea 上的**持久 Claude Code 会话**通信：

```
用户(微信) ⇄ Hermes(spark) --ssh korea--> claude --resume <会话ID> ⇄ kcc 的上下文记忆
```

## 通道规格

| 项目 | 值 |
|------|-----|
| 会话 ID | `9db3280e-7927-4c54-bb74-bcee66d1064b` |
| 工作目录 | `/home/ubuntu/workspace/claw_bot/test/` |
| 建立时间 | 2026-08-26 09:24 CST |
| 上下文文件 | `~/.claude/projects/-home-ubuntu-workspace-claw-bot-test/9db3280e-7927-4c54-bb74-bcee66d1064b.jsonl` |

## 发送方式

```bash
ssh korea 'cd /home/ubuntu/workspace/claw_bot/test && claude -p --resume 9db3280e-7927-4c54-bb74-bcee66d1064b "<消息>" --max-turns 1 --effort high'
```

- 轻量问答 / 闲聊：`--max-turns 1`，秒级到分钟级返回
- 重活（多步任务）：另起新会话 + nohup 后台跑，跑完把结果带回
- 记忆已验证：kcc 能复述线程里上一轮内容（2026-08-26 实测 ✅）

## 边界

- kcc 只在 `test/` 目录内操作（`CLAUDE.md` 软约束）
- 通道不承载敏感凭据；密钥/token 一律不进对话内容
- 若会话日后失效，重建新会话后更新本文件
