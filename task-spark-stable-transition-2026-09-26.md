# 任务记录：让 spark 成为稳定的过渡段

- **日期**：2026-09-26
- **状态**：新登记

## 任务内容

让 spark (本地工作站 VLLM 服务，端口 8000) 成为模型切换和任务过渡的**稳定中段**：确保在主模型（目前为 Nemotron3.5-Lightning-30B-A3B）与备用/替代模型之间切换时，spark 能够可靠、无数据丢失地承中间过渡角色，保证任务执行的连续性和模型服务的平滑过渡。

## 详细需求

1. **切换鲁棒性**
   - 模型切换操作（如 qwen3.6 → qwen3.8 切换脚本）在 spark 上执行时，若中断或失败，能自动回滚到上一个可用模型，而非留下半加载或错误状态的模型
   - 切换前后的服务可用性至少 99%（即切换过程不应导致长时间服务不可达）

2. **配置持久化**
   - 切换完成后，hermes config.yaml (`~/.hermes/config.yaml`) 中的 `model.default` 和 `providers.custom.base_url` 应自动同步更新为新模型名，无需手动 `sed` 替换
   - 共享下游服务（hermes-gateway、common-api on Tencent）的模型引用也应相应更新，避免指向失效的模型名

3. ** smoke 测试自动化**
   - 切换后自动进行最少 2 个 smoke test：
     a. 纯文本聊天测试（`max_tokens: 100`，`temperature: 0`）
     b. 工具调用测试（含 function get_weather 的调用）
   - 两个 test 均通过（JSON 响应中模型 ID 匹配、无错误）才标记切换成功

4. **日志与溯源**
   - 所有切换操作记录到 `~/models/Scripts/cutover_*.log`，包含：开始时间、旧模型名、新模型名、容器 ID、就绪耗时、 smoke test 结果
   - 若切换失败，日志中需明确标记回滚操作和原因

5. ** spark 资源监控**
   - 切换过程中监控 spark 容器的 CPU/内存使用，确保未超过阈值（CPU < 80%，内存 < 85%）导致的切换失败
   - 若检测到资源压力过大，应在切换前发出警告并暂停切换

## 备注

- 灵感来源：现有的 `cutover_qwen38.sh` 脚本已实现原子化容器切换，但需要手动同步配置且无自动 smoke test
- 本任务目标是固化为可复用的流程，未来每次模型迭代都能按此标准执行
- 当前 spark 上运行：`vllm-nemotron35-nvfp4` (nemotron3.5-lightning-30b-a3b) 端口 8000
- 备选模型：`qwen3.8-27b-fp8` (端口 8000 同一容器复用) / `qwen3-asr-1.7B` (端口 8001)