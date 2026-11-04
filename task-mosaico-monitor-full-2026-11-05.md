# 任务记录：Mosaico 监控网页完整版

- **日期**：2026-11-05
- **状态**：待开发
- **要求**：用户不用再描述板子上的进展和界面，打开网页（或让 agent 读接口）就能看到真实画面、当前应用、异常和结果

## 任务内容
把已有的 Mosaico 监控网页升级成完整版：真实屏幕截图、内置应用也可见、板子心跳和重启检测、异常自动判断并推送、各应用结果汇总。

> 给执行这个任务的 agent：假设你对 Mosaico 一无所知。请先完整读完本页，再按"第 0 步"熟悉环境，最后按阶段动手。拿不准、或者要做本页没写到的危险操作时，先问用户。

## 一、先验知识：Mosaico 是什么

### 1. 硬件

- **ESP-Mosaico** 是乐鑫（Espressif）出的一块桌面开发板：
  - 芯片 ESP32-S31，屏幕是 480×480 的**圆形**触摸屏
  - 带麦克风、喇叭、IMU、磁力计、摄像头接口，有电池
- 官网 https://mosaico.espressif.com/ ，固件源码 https://github.com/esp-mosaico/esp-mosaico-claw （公开）。
- 用户只有一块板子，平时在家，也经常用手机热点上网（网速只有约 100 KB/s）。所以板子大多数时候**不在家庭局域网里，你从外部连不到它**，只能等它主动发请求出来。

### 2. 固件：Claw

板子跑的是官方的 **Claw 固件 v0.5.0**，上面打了用户自己的小补丁：主屏多了一个 Voice Chat 图标。界面分两类：

| 类型 | 例子 | 技术 | 能不能改 |
|---|---|---|---|
| **内置应用** | 主屏（Hub/Launcher）、天气、设置、相册、音乐、Works | C 语言 + 乐鑫的 GSP 场景引擎（`components/mosaic_ui/`），不是普通的 LVGL 控件树 | 只能改固件再刷机 |
| **Lua 应用** | 本项目自己写的 6 个应用 | 跑在 Works 里的 Lua 脚本，用 Lua 版 LVGL 画界面 | 发布到应用商店，用户在板子上重装即可，不用刷机 |

**Lua 应用的一生：**
1. 我们在 VM 上写好。
2. 用 `publish_store.py` 发布到自建商店 **My Store**（`https://www.wangyutang.com/mosaico-store`）。
3. 用户在板子上依次进入 Works → My Store → Install。
4. 更新时，用户要先 Uninstall 再 Install，商店不会覆盖安装。每个应用界面上都显示"更新 <时间>"，用来确认装的是不是新版。

### 3. 机器和网络

| 机器 | 作用 | 怎么访问 |
|---|---|---|
| **VM**（你就在这台上） | 代码所在地；跑 Caddy（`www.wangyutang.com` 的 HTTPS 入口）、应用商店静态文件、监控服务 | 本机，用户名 `ubuntu`，有 sudo，时区 Asia/Shanghai |
| **spark** | 用户家里的 GPU 服务器（aarch64）；跑语音中继（语音识别 → 大模型 → 语音合成）和听声识鸟服务 | `ssh spark`（VM 上已配置好）。VM 通过 ssh 隧道把 spark 的服务转到本机端口，再由 Caddy 对外 |
| **WSL** | 用户 Windows 电脑里的 Ubuntu；板子的 USB 线插在这台电脑上，**固件只能在这里编译和刷机** | `ssh wsl`（经过反向隧道；电脑睡眠时连不上） |

**截至 2026-11-05：spark 和 WSL 都离线。**
- 第一阶段不依赖这两台机器。
- 第二阶段需要 WSL 在线，并且用户在电脑旁边。

### 4. 公网入口（都在 VM 的 Caddy 上，配置文件 `/etc/caddy/Caddyfile`）

| 路径 | 后端 | 鉴权 |
|---|---|---|
| `/mosaico-store/` | 静态目录 `/srv/mosaico-store` | 无（公开） |
| `/mosaico-chat/` | 语音中继（经隧道到 spark） | Bearer token |
| `/mosaico-bird/` | 听声识鸟（经隧道到 spark） | Bearer token |
| `/mosaico-pokedex/` | Pokedex 素材 | Bearer token |
| `/mosaico-monitor/` | 本任务的监控服务，`127.0.0.1:8095` | 页面要 basic auth；`/v1/*` 要 Bearer token |

- 所有 Bearer 接口共用**同一个中继 token**，存放位置有：
  - spark 的 `~/.config/mosaico-chat-relay/token`
  - 板子上的 `/nand/config/voice_chat.json`（`token` 字段）
  - VM 的 `/etc/caddy/pokedex_auth.caddy`（只有 root 和 caddy 用户能读，Caddy 用它来校验）
- 监控页面的 basic auth 用户名写在 `mosaico-monitor/README.md`，密码存在 VM 的 `~/.config/mosaico-monitor/viewer_password`。

### 5. 代码在哪

VM 上的 `~/workspace/aicoding/ai_mosaico/` 是一个 git 仓库（分支 `main`），每个项目一个目录：

| 目录 | 内容 |
|---|---|
| `mosaico-monitor/` | **本任务要改的监控服务**：`monitor.py`、`index.html`、`mosaico-monitor.service`、`README.md` |
| `claw-voice-chat/` | 语音助手 Lua 应用；`firmware/0001-hub-voice-chat-icon.patch` 是主屏图标的固件补丁，**改固件时照着它的写法来** |
| `claw-hello/`、`claw-tap-compass/`、`claw-pokedex/`、`claw-birdsong/` | 其他 Lua 应用，都已接入监控上报 |
| `claw-my-store/` | 应用商店和发布脚本（`apps.txt` 列出要发布的应用） |
| `chat-relay/` | spark 上的语音中继 |
| `.claude/skills/mosaico-ops/` | **运维手册**：`SKILL.md`（主机、固件状态、安全规则），`references/claw-firmware.md`（编译刷机），`claw-lua-apps.md`（Lua 应用写法和踩坑），`spark-services.md`、`wsl-usb.md` |

## 二、第 0 步：先熟悉环境（不要跳过）

1. 读这些文件：
   - `mosaico-monitor/README.md` 和 `monitor.py`、`index.html` 全文
   - `.claude/skills/mosaico-ops/SKILL.md`
   - `references/claw-lua-apps.md`
   - 任意一个应用的 `report` / `flush_report` 代码块，例如 `claw-hello/hello_mosaico/scripts/hello.lua`
2. 看第一版正在跑的状态：
   ```bash
   systemctl status mosaico-monitor
   curl -s 127.0.0.1:8095/api/state | python3 -m json.tool | less
   ```
   `api/state` 里有 `board`（最后一次看到板子）、`spark`（后端健康）、`current`（当前应用）、`apps`（各应用版本）、`alerts`（告警）、`timeline`（时间线）。
3. 浏览器打开 `https://www.wangyutang.com/mosaico-monitor/`，看第一版长什么样。密码见上文。
4. 看 Caddy 里 `mosaico-monitor` 那一段配置：`sudo grep -n -A25 'mosaico-monitor' /etc/caddy/Caddyfile`。**不要打印 `pokedex_auth.caddy` 的内容**。
5. 改固件之前，把 Claw 源码浅克隆到临时目录读一读：`git clone --depth 1 https://github.com/esp-mosaico/esp-mosaico-claw`。注意：
   - 板子上的版本是 v0.5.0（提交 `2e1017e`），最终要改的是 WSL 上的分支，见第二阶段；
   - 公开仓库的 main 比 v0.5.0 新，读代码时以 v0.5.0 为准。

## 三、理想效果（验收以此为准）

1. **看到真实画面**：
   - 网页中间的圆形区域显示板子屏幕的真实截图；
   - 画面变化时更新，没变化也每 30 秒更新一张；
   - 可以往回翻最近的截图，每张带时间和所在应用。
2. **所有界面都可见**：
   - 主屏、天气、设置等内置界面和商店应用一样，显示"当前在哪个应用、什么时候进入"；
   - 时间线记录完整的切换过程，例如：主屏 → 天气 → 主屏 → Pokedex（皮卡丘）→ 退出。
3. **板子状态**：板子每 30 秒发一次心跳，网页显示：
   - 在线与否、上次心跳时间
   - 开机时长，上次重启的时间和原因
   - Wi-Fi 信号、剩余内存、电量（电量在固件能拿到时才显示）
4. **异常自动判断**，标红并推送，恢复后自动标为"已解决"：

   | 异常 | 判断依据 |
   |---|---|
   | 应用崩溃或报错 | 应用上报 `error` / `fatal` 事件 |
   | 卡死 | 有心跳，但画面和触摸 3 分钟没变化，且最后一条事件是"加载中"类状态 |
   | 黑屏或白屏 | 截图几乎是纯色 |
   | 掉线 | 心跳中断超过 2 分钟 |
   | 重启 | 开机时长变小，或重启原因不是正常上电 |
   | 旧版本 | 板子上应用的版本早于商店版本（第一版已有） |
   | 后端不可用 | spark 上的语音、识鸟服务健康检查失败（第一版已有） |
   | 内存不足 | 剩余内存低于阈值（按实测定） |

5. **应用结果汇总**：每个应用最近一次的结果，例如听声识鸟识别出的鸟和置信度、语音助手的问答、Pokedex 看了哪一只。
6. **给 agent 用的接口**：状态 JSON 和最新截图都能在 VM 本机用 `curl` 拿到，agent 回答"现在怎么样"之前先读接口。

## 四、现状（第一版，2026-11-04 上线）

- **`monitor.py`**：Python 标准库实现，不依赖第三方包。
  - 监听 `127.0.0.1:8095`，数据存在 SQLite `~/mosaico-monitor/monitor.db`。
  - 由 systemd 系统单元 `mosaico-monitor` 运行，用户 `ubuntu`，附加组 `adm`，用来读 Caddy 日志。
  - 数据来源有三个：
    1. 跟随 `journalctl -u caddy -f` 读 Caddy 访问日志，按 User-Agent `ESP32 HTTP Client` 挑出板子的请求，用来判断板子是否在线、装了什么、请求有没有失败（比如 404、502）。
    2. 每 60 秒检查一次 spark 上两个服务的 `/health`。
    3. Lua 应用调用 `POST /v1/event` 上报事件 `{app, ver, kind, msg, data}`。
       - `kind` 取值：open / screen / result / error / fatal / exit；
       - `msg` 是"屏幕上现在显示什么"的一句中文；
       - `ver` 是应用的更新时间。
- **`index.html`**：中文仪表盘，每 5 秒拉一次 `api/state`；屏幕区域现在是根据文字画的模拟图。
- **应用端**：6 个 Lua 应用都接入了上报。事件先在回调里排队，再由主循环逐条发出（超时 3 秒），token 从板子上的 `/nand/config/voice_chat.json` 读取。

**缺口**：
- 没有真实截图；
- 内置应用不经过 Lua，所以完全看不到；
- 没有心跳，"在线"只能靠板子偶尔发出的请求来猜；
- 异常只靠应用自己报；
- 没有推送。

## 五、第一阶段：只改服务器，不用刷机（现在就能做）

1. **新接口**，沿用 `/v1/*` 的 token 鉴权：
   - `POST /v1/heartbeat`：JSON，字段包括 `uptime_s`、`reset_reason`、`free_heap`、`min_free_heap`、`rssi`、`battery`、`current_app`、`fw`。字段名可以调整，但要写进 README，第二阶段的固件按这个实现。
   - `POST /v1/screenshot`：请求体是 JPEG 二进制，`Content-Type: image/jpeg`，用 `X-App` 头带上当前应用。
     - 截图存成文件，数据库只存索引；
     - 保留 24 小时，最多 500 张；
     - Caddy 里给这个接口单独放宽请求体上限（建议 256KB），其他接口保持 32KB。
   - `POST /v1/event` 增加 `kind: "switch"`，用于固件上报应用切换。
   - 只在本机开放 `GET /api/screenshot/latest` 和 `GET /api/screenshot/<id>`，网页和 agent 都用。
2. **异常规则**：实现第三节表格里的全部规则，包括截图纯色检测和"卡死"判断，阈值集中写在配置里。
   - 每条规则写单元测试（`python3 -m unittest`）；
   - 告警恢复的逻辑沿用第一版：后面出现成功的事件，就把之前的告警标为已解决。
3. **推送**：做成可配置的 webhook，地址和密钥放在 `~/.config/mosaico-monitor/` 下，不进仓库。
   - 推送渠道还没定（见第八节），先实现浏览器内提醒（标签页标题闪动 + 声音，可关闭）；
   - 同一条告警 30 分钟内只推一次。
4. **网页改版**：
   - 真实截图区，附截图历史，可以翻页；
   - 板子状态卡片；
   - 应用结果汇总；
   - 时间线加入应用切换；
   - 没有截图时，退回第一版的文字模拟屏；
   - 保持中文、浅色和深色主题、手机上能看。
5. **模拟板子**：写 `mosaico-monitor/fake_board.py`，用真实 token 向 `https://www.wangyutang.com/mosaico-monitor/v1/...` 发心跳、截图、切换和各种异常场景。
   - token 用 `sudo` 从 `pokedex_auth.caddy` 里取出来直接作为变量用，**不要打印**；
   - 发完要清掉这些测试数据，不然会污染真实记录。第一版就出过这个问题：测试事件覆盖了板子的 IP。
6. **部署**：
   - 改完执行 `sudo systemctl restart mosaico-monitor`；
   - 改 Caddy 前先备份成 `/etc/caddy/Caddyfile.bak.<日期>_<原因>`，用 `sudo caddy validate --config /etc/caddy/Caddyfile` 校验通过后再 `sudo systemctl reload caddy`。

## 六、第二阶段：改 Claw 固件（需要刷机）

### 要加的功能

在固件里新增一个监控模块：

1. **应用切换上报**：
   - 位置：`components/mosaic_ui/core/mosaic_esp_platform.c` 的 `platform_open_app` / `platform_close_app`，以及回到主屏的地方；内置应用的打开和关闭都经过这里。
   - Lua 应用由 Works 和 `components/mosaic_ui/common/works_runtime.c` 启动，那里也要加钩子。
   - 上报 `{app, kind: "switch"}`，Lua 应用的名字要和它自己上报用的 id 一致（如 `pokedex`、`bird_id`）。
2. **心跳**：每 30 秒一次。
   - 重启原因用 `esp_reset_reason()`；
   - 内存用 `heap_caps_get_free_size`；
   - Wi-Fi 信号用 `esp_wifi_sta_get_ap_info`。
3. **截图**：
   - **注意**：内置界面由 GSP 引擎直接出帧，不是 LVGL 控件树，所以 `lv_snapshot` 只能截到 Lua 应用。要拿完整画面，应该在 `components/display_service` / `components/esp_lvgl_present` 的出帧环节复制最终的帧缓冲。先读懂这里的代码，再定方案。
   - 编码：固件已经依赖 `espressif/esp_new_jpeg`，缩到 240×240、质量调到每张 30KB 以内。
   - 时机：画面有刷新且距离上一张超过 3 秒，或者已经 30 秒没传。
   - 必须放在独立的低优先级任务里，不能阻塞 UI 和音频；内存不够时宁可跳过这一张。
4. **网络失败不能影响板子正常使用**：
   - 超时设短（3 秒）；
   - 失败就丢弃，不排队重试；
   - 没有 Wi-Fi 时什么都不做。
5. **token 从板子上读取**：读 `/nand/config/voice_chat.json` 的 `token` 字段（文件在 FAT 分区上，路径以固件里的挂载点为准），不编译进固件。
6. **顺带做"My Store 放到主屏"**，方式见第八节第 4 条：
   - 写法参考 `claw-voice-chat/firmware/0001-hub-voice-chat-icon.patch` 和 `references/claw-firmware.md`；
   - 主屏上的文字只能用 ASCII，中文会显示成 `????`。

### 编译

- 在 WSL 上：源码 `~/mosaico/esp-mosaico-claw`，分支 `voice-chat-home`（基于 v0.5.0 的 `2e1017e`，已有图标补丁），ESP-IDF 用 `~/esp/esp-idf-claw`。
- 具体步骤见 `references/claw-firmware.md` 和 `references/wsl-usb.md`。要点：
  1. 先 `source` 环境脚本，再 `export IDF_PATH=~/esp/esp-idf-claw`。
  2. 用 `ninja -C build -j 4` 编译：WSL 只有约 7GB 内存，并行开满会被系统杀掉。
  3. 用 `setsid nohup` 在后台跑，因为 ssh 断开会中断编译。全量编译约 20–30 分钟。
- WSL 上没有配置 git 身份，提交时加上 `git -c user.name=... -c user.email=...`。
- 改完后执行 `git diff --binary 2e1017e HEAD`，把补丁导出到 `ai_mosaico` 仓库，放在 `mosaico-monitor/firmware/` 下。

### 刷机：必须先征得用户同意

1. 告诉用户要刷机，确认用户就在电脑旁边，并且同意。
2. 用户把板子用 USB 接到 Windows 电脑，关机后按住 BOOT 再点一下 POWER，进入下载模式（黑屏）。
3. 用 usbipd 把 USB 设备挂到 WSL，见 `wsl-usb.md`。
4. 在 WSL 上运行 `idf.py -p /dev/ttyACM0 update-flash`：
   - 它只写应用分区，Wi-Fi、配置和已安装的 Lua 应用都会保留；
   - 大约需要 7 分钟。
5. 刷完板子会自动重启进入 Claw，然后把 USB 交还给 Windows。

## 七、约束与安全（必须遵守）

- **刷机**：
  - 只能用 `update-flash`；
  - 禁止 `idf.py flash`、直接用 esptool 写、整片擦除，这些会清掉用户的数据；
  - 不要从 Windows 打开 Claw 的串口：打开时 DTR/RTS 信号会让板子复位，或者进入下载模式。
- **密钥**：中继 token 和监控密码不能打印，也不能写进聊天、git 提交、本网站或应用商店（商店是公开的）。推送代码前，用实际值 `grep -F` 扫一遍，确认没有。
- **隐私**：截图只存在 VM 上，只能登录后查看；不要把截图发到任何第三方服务。
- **spark**：用户测试期间不要重启 spark 上的服务，否则正在进行的语音会中断。
- **Lua 应用**：如果要改 Lua 应用，按 `claw-lua-apps.md` 的规定来：
  - 每个应用显示更新时间；
  - 都要上报监控；
  - 重新发布后提醒用户先 Uninstall 再 Install。
- **用户沟通**：用户使用简体中文；不要问用户"屏幕上显示什么"，先自己读监控。
- **提交**：所有改动都提交到 `ai_mosaico` 仓库，并更新 `mosaico-monitor/README.md` 和 `mosaico-ops` 技能里的相关说明。

## 八、待用户确认（开工前先问）

1. **截图频率**：画面变化时 + 每 30 秒一张，可以吗？
2. **推送渠道**：严重异常推到哪里？没定之前只做网页内提醒。
3. **截图保留**：保留 24 小时，最多 500 张，可以吗？
4. **主屏入口**：第二阶段顺带做"My Store 放到主屏"，选哪种方式？
   - **快捷入口**：主屏上放一个通用的快捷入口，它读取 `/nand/config/home_shortcut.json` 决定打开哪个应用，以后换应用只要上传这个文件；
   - **独立图标**：直接多加一个 My Store 图标。

## 九、验收标准

1. **第一阶段**：用 `fake_board.py` 跑一遍，要求：
   - 网页能显示截图、心跳和内置应用的切换；
   - 第三节表格里每类异常都能触发，也都能自动恢复；
   - 单元测试全部通过；
   - 测试数据已经清理干净。
2. **第二阶段**：刷机后，在板子上依次打开 主屏 → 天气 → Pokedex → 退出 → 听声识鸟，要求：
   - 网页时间线顺序正确；
   - 截图和实物一致，延迟不超过 10 秒；
   - 关掉 Wi-Fi 后 2 分钟内报掉线，恢复后告警自动解决；
   - 板子操作没有明显卡顿，声音正常。
3. **agent 自查**：只读接口就能准确说出板子当前的界面和最近的异常，不需要问用户。
4. **交付**：给用户写一份简短的中文总结，包括：做了什么、怎么看、已知问题、还没做的部分。
