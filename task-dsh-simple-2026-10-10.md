# 任务记录：简单 dsh 实现
- **日期**：2026-10-10
- **状态**：今日任务

## 任务内容
实现一个简单的 dsh（Distributed Shell）工具，用于通过 hostfile 将命令并行分发到多台服务器。

### 核心功能
1. **hostfile 格式**：支持 `hosts.txt` 文件，每行一个 hostname，支持 `user@host` 格式
2. **Parallel command execution**：并行执行命令，显示每台主机的输出
3. **SSH 连接管理**：使用 SSH with ConnectTimeout 和 BatchMode
4. **输出格式化**： `[hostname]` 前缀标识输出来源

### 实现方式
- Bash 脚本使用 `getopts` 解析 `-f <hostfile>` 和 `-c <command>` 参数
- 逐行读取 hostfile，通过 SSH 远程执行命令
- 支持 `user@host` 格式提取用户名
- SSH ConnectTimeout 10秒，BatchMode yes 避免卡顿

### 使用示例
```bash
# 创建 hostfile (hosts.txt)
# host1.example.com
# user2@host2.example.com
# host3.example.com

# 执行命令
dsh -f hosts.txt -c "uname -a"

# 期望输出
# [host1.example.com] Linux host1 5.15.0-...
# [host2.example.com] Linux host2 5.15.0-...
# [host3.example.com] Linux host3 5.15.0-...
```

### Hostfile 格式说明
- 每行一个主机地址
- 直接写 hostname：`host1.example.com`
- 写用户前缀：`user@host.example.com`
- 空行和以 `#` 开头的行会被跳过

### 注意事项
- 目标主机需要配置 SSH 密钥免登录
- 超时时间默认 10 秒，可根据网络情况调整
- 串行处理每台主机，但可通过 & 后台运行实现真正并行（高级功能）