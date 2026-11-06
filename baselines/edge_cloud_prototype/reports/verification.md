# 验证记录：四模式与简化DoT真实联调

2026-10-02，沿用原AutoDL小模型，新增dot并接入用户指定的DeepSeek `deepseek-flash`（`thinking.type=disabled`）。短生成、四组真实对照及演示CLI均已运行。下面“本轮”是当前证据；后面的12题小模型数据属于早期历史基线，不能与新增6题直接比较。

## 本轮：2026-10-02四组真实联调

实验目录：[dot-verified-20261002](dot-verified-20261002/)。实验完成UTC时间 `2026-10-02T14:00:42.882342+00:00`，状态complete。密钥只在实例 `runtime/secrets/deepseek_api_key` 读取，权限600；本地和ZIP不含该文件。模型与核心代码、题集的哈希已随protocol保存，并在实验后再次逐项核对一致。

- 原实例仍运行，恢复既有服务后 `/health=ok`；没有新增租赁或重复下载。运行时observed wrapper PID1515、llama-server PID1516、GPU约3320MiB，这些为本轮观察值，不是固定配置。
- 两端doctor均ready。云端短请求“6加7”真实返回13，输入26、输出1、总27 tokens；完整usage检查通过后才开始对照。该请求及一次本地预热单独留档。
- 新增题集6题，SHA256 `dd9f8cb480e8ae8d7a3ba0ae90a8b0a16518a9fd3af1eca8f7f5902354479638`。参考答案经整数算术与组合穷举核对；固定哈希测试能发现“改题干但保留答案”。原12题未改。
- AutoDL Linux最终37项测试全部通过，日志见 `dot-verified-20261002/unit-tests.log`；Windows37项中35通过、2项按Linux CMake/POSIX权限条件跳过。两端compileall通过。HTTP测试覆盖正常顺序/前置结果、无效计划和边界、本地仅升级一次、云端失败不重试、预算超限、阶段及未知usage、旧模式回归、禁思考参数和ID关联。

| 模式 | 正确 / 总数 | 成功率 | 云端tokens | 平均ms | P50 ms | P95 ms | 升级 / 退化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 1 / 6 | 100% | 0 | 192.22 | 78.80 | 491.38 | 0 / 0 |
| cloud | 4 / 6 | 100% | 391 | 658.11 | 676.10 | 769.61 | 0 / 0 |
| hybrid | 2 / 6 | 100% | 233 | 436.61 | 454.52 | 828.59 | 0 / 0 |
| dot | 4 / 6 | 100% | 2155 | 1851.73 | 1864.21 | 2244.80 | 0 / 0 |

DoT分解6次/1012云端tokens/5085.65ms，汇总6次/1143云端tokens/4265.25ms；中间20次均由规则分给local，本地3291 tokens/1751.92ms。阶段耗时是所有该阶段调用耗时之和；请求elapsed_ms是完整墙钟时间，含调度开销。旧stage elapsed_ms字段仅表示云端调用耗时，all_call_elapsed_ms包含本地调用。

本轮24请求共有50次模型调用，云端21次、2779 tokens。独立短生成27 tokens；实验后验证README中的演示命令一次，返回146，额外云端2次/446 tokens，完整2199.39ms。本轮合计24次云端调用、3252云端tokens，usage无缺失。数值不是账单金额，不含此前历史调用。

独立审计记录 [audit.json](dot-verified-20261002/audit.json) 核对了源码/数据哈希、24个题目模式配对、唯一ID、严格评分、原始calls与各阶段/总量、平均/P50/P95及时延覆盖关系。原始错误答案全部保留：dot的d04选A而正确为B，d06选BC而正确为AB。本轮没有节省token或加速的证据；没有事后修改评分或重跑挑选结果。

计时客户端位于AutoDL；排除启动和SSH传输，预检/预热单列；应用不缓存，服务/供应商缓存未控制。6题单轮仅作流程演示。老师说明、真实示例和实际运行命令见 [TEACHER_DEMO.md](../TEACHER_DEMO.md)。交付ZIP白名单含源码、测试、非敏感配置、文档和选定公开实验记录，排除runtime、模型、虚拟环境、实际敏感配置和个人材料；SHA以包外sidecar为准。

## 历史阶段：初次部署与12题本地基线

下列记录发生于云端尚未配置时，保留原始测试数字和部署依据。

## 实例与服务

| 项目 | 实际配置 |
|---|---|
| 地区 / 主机 | 西北 B 区 / 113 机，主机 ID `6fcb40b7f5` |
| 实例 ID | `6fcb40b7f5-2694e599` |
| GPU | RTX 3080 Ti 12GB ×1；SSH 实查 12288MiB |
| CPU / 内存 | 分配 12 核 / 90GB |
| 磁盘 | 30GB 系统盘 / 50GB 免费数据盘 |
| 镜像 | PyTorch 2.8.0 / Python3.12 / Ubuntu22.04 / CUDA12.8 |
| 实查环境 | Python3.12.3、驱动580.76.05、CUDA toolkit12.8.93 |
| 计费 | 按量创建报价 1.14元/小时；运行中持续计费 |
| 项目目录 | `/root/autodl-tmp/edge-cloud-prototype/prototype` |
| 接口 / 模型别名 | 实例内 `http://127.0.0.1:8000/v1` / `qwen-local` |

实例内完成 CUDA 源码编译，启动日志确认 CUDA ARCHS 860 和 **37/37 层加载到 GPU**。验收时 `llama-server` PID 10030 使用3324MiB显存，整卡观察值3333MiB；后台启动包装进程 PID10029。这些 PID 和占用是验收时观察值，重新启动后会变化。部署 SSH 连接结束后重新连接，健康接口仍为 `ok`，真实请求 `6乘以7等于多少？只输出整数。` 返回 `42`。

初次部署时 `doctor.local.status=ready`、`doctor.cloud.status=not_configured`，当时总退出码2表示云端缺配置。当前四模式入口两端均ready。使用免费数据盘；用完实例应在控制台关机，价格与磁盘计费以平台显示为准。

## 模型和运行时校验

- 模型：`Qwen/Qwen3-4B-GGUF` / `Qwen3-4B-Q4_K_M.gguf`。
- 固定模型 revision：`bc640142c66e1fdd12af0bd68f40445458f3869b`。
- 完整文件大小2497280256字节，实测SHA256与发布方一致：`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`。
- llama.cpp：release `b11321` 对应固定源码commit `b0aca3c6539e2dd55ea510bbb79591852a4d4b81`，在实际Ubuntu22.04镜像内编译CUDA服务。
- 官方源码归档37755513字节，SHA256 `b5475a493e0d17f8af121a4a442fdda78fe0ef6f19f230a5c1690b699ee49c95`。此哈希是官方HTTPS下载的观察值，未声称发布方独立提供。
- 二进制实际自报 `0.5.0-dev (build 0, commit unknown)`，GNU11.4.0 / Linux x86_64。源码归档没有Git工作树，版本依据为固定源码commit和归档SHA，而非二进制自报release。

下载优先镜像；遇到下载中断后使用官方加速源16MiB分段续传，合并完整文件后再验证整体SHA256。模型校验与部署元数据见 `local-verified-20261002/deployment-evidence.json`，固定下载清单见 `../scripts/deployment-manifest.json`。

## 真实本地基线

原始记录目录 `local-verified-20261002/`，部署证据时间为UTC `2026-10-02T07:58:27.820381+00:00`。`simulated=false`。

| 指标 | 实测 |
|---|---|
| 题数 / 请求数 | 12 / 12，重复次数1 |
| 成功返回 / 请求错误 | 12 / 0 |
| 严格准确率 | **6/12 = 50%** |
| 平均完整响应 | 179.84ms |
| P50 / P95 | 58.83ms / 455.40ms |
| 外部云端调用 / 云端tokens | 0 / 0 |

首题 `17加25等于多少？只输出整数。` 返回纯 `42`，完整响应83.51ms；接口报告本地输入42、输出3、总45tokens。上述45为本地模型tokens，不计云端用量。

评分保持预先固定的完整答案匹配规则，没有事后改评分。未命中的6题中，s06排序返回 `3,1,2`、c03串行任务耗时返回 `5`，这2题实际答错；c01/c02/c05/c06附带解释，未遵守只输出整数的要求，4题按严格匹配计错。原始回答见 `requests.jsonl`。

该12题公开合成冒烟集用于验收请求、计量及报告流程，50%不能解释成模型质量达标。客户端在实例内访问同一热服务；应用不缓存答案，但llama.cpp实际复用了前缀KV缓存。报告 `cache_enabled:false` 仅指应用缓存关闭。时延未包含模型启动或Windows SSH传输时间，样本量不足以得出稳定P95结论。

## 代码修正与现有检查

- 固定源码中server目标在tools条件内，真实构建曾报 `No rule to make target 'llama-server'`。部署脚本将 `LLAMA_BUILD_TOOLS` 改为ON，并新增Linux CMake目标注册回归检查。
- 单独设置思考预算0仍产生思考标记或截断输出；启动脚本增加 `--chat-template-kwargs '{"enable_thinking": false}'`，保留 `--jinja --reasoning-budget 0`，并用日志级别4记录GPU加载。早期失败基线保留于实例 `reports/local-first-20261002/`，最终基线使用新目录，没有覆盖。
- AutoDL Linux：23项现有单元测试全部通过，回传日志 `local-verified-20261002/unit-tests.log`。覆盖接口、认证、三模式、仅升级一次、未知usage、截断、严格评分、不可覆盖报告、模拟标记、SHA与归档路径、CUDA构建错误传播。
- 最终Windows检查：`python -X utf8 -m unittest discover -s tests -v`，23项中22通过，1项Linux CMake检查按平台跳过；`python -X utf8 -m compileall -q edge_cloud scripts` 通过。
- 先前CLI模拟演示为12题×3模式，明确 `simulated:true`、token未知；模拟结果不属于真实模型实验。

## 使用与证据交付

在实例“JupyterLab”中打开Terminal，执行：

```bash
cd /root/autodl-tmp/edge-cloud-prototype/prototype
/root/miniconda3/bin/python -m edge_cloud ask '17加25等于多少？只输出整数。' --mode local
```

当前服务在后台运行。实例关机再开机后，先在一个终端执行 `/root/miniconda3/bin/python scripts/start_autodl.py` 并保持该终端打开，然后在另一个终端运行ask。

真实证据包含 `requests.jsonl`、`summary.json`、`first-ask.json`、`doctor.json`、`deployment-evidence.json`、`unit-tests.log`。原始回传ZIP `local-verified-20261002.zip` 为5769字节，SHA256 `20841cb992b3ad2f1de7d53ed12e4cf45cf2034249298a41de7b7cd0932f3bbe`。

更新后的 `../autodl-prototype.zip` 包含代码、部署脚本、测试、文档及上述6个真实证据文件；路径统一带 `prototype/` 前缀。不含模型、runtime、虚拟环境、密钥、实际配置或原始申请材料。包的SHA256记录在同目录sidecar，避免包内自引用。

## 当前未验证与后续阶段

外部API、真实usage和6题四组对照已完成，结果见本文开头；尚未证明性能或成本收益。GSM8K/MMLU、阈值扫描、工具执行、多轮讨论及隐私模块均未完成。正式实验还须控制缓存、增加题集与重复次数，并与老师确定指标阈值。使用步骤见 `../README.md`。
