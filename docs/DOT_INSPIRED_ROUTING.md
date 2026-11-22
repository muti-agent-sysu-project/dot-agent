# DoT 启发的端云编排（2026-11-22）

本页记录从 `dot_replicate/DoT`（WWW'25 *Division-of-Thoughts*）吸收到
`dot_agent` 主流水线的设计，以及在本地 Qwen3-4B-Instruct-2507 上的实测结果。

## 吸收了 DoT 的哪些设计

| DoT 原始设计 | 本仓库落点 | 作用 |
|---|---|---|
| Task Decomposer：把原题拆成可求解子问题 | `orchestrator/decomposer.py` | 默认本地分解；`DOT_DECOMPOSE_TIER=cloud` 可切云端 |
| 子问题求解时始终携带原题与已完成答案 | `pipeline.py` 的 executor prompt | 避免 4B 把子问题当成脱离原题的孤立问句 |
| Task Scheduler：判断子问题依赖并建 DAG | `orchestrator/scheduler.py` | 依赖判断默认交给云端，图更完整；本地兜底 |
| 识别关键步骤并按难度分配强模型 | `TaskScheduler.key_steps()` + executor | DAG 汇点（无后继）直接交云端；普通中间步骤留本地 |
| 最终汇总只输出最终答案 | `pipeline.py` 的 summarize | 单汇点时直接采用关键步骤答案；多汇点才汇总 |
| 依赖/汇总/关键步骤可由云端承担，子任务本地执行 | 端云分工 | 4B 负责便宜的子步骤，云端负责规划与关键答案 |

关键实现点：

1. **原题上下文永不丢弃。** 每个子任务 prompt 都带 `原问题: ...`，已完成答案
   按预算注入，直接前驱标注“重点参考”。
2. **汇点 = 关键步骤。** `key_steps()` 只把有效 DAG 的汇点交给云端；有环、
   解析失败或无效边时返回空集，退化为普通信号路由，避免整题误判上云。
3. **最终答案取唯一关键步骤。** REASONING 只有一个汇点且答案含数字时，直接
   采用该答案，不再做一次可能截断/抄错的本地汇总。
4. **汇总双档回退。** 需要汇总时 REASONING 云端优先（独立复核），其他类型
   本地优先；两端都失败才回退最后一个子任务答案，不再抛整题失败。
5. **依赖判断走云端。** 云端不可用时 `MeteredClient` 自动回退本地，保持离线可用。
6. **分解默认本地。** 云端分解可切换（`DOT_DECOMPOSE_TIER=cloud`），但 2026-11-22
   对照显示没有准确率收益、云端 token 约翻倍，因此默认不做。分解模型异常
   （截断/空答）重试一次，仍失败返回 `None`，由整题云端回退。

## 实测结果（2026-11-22）

环境：llama.cpp `qwen-local`（Qwen3-4B-Instruct-2507-Q4_K_M，4096 ctx，
RTX 5060 Laptop 8GB）+ DeepSeek `deepseek-flash`（`thinking: disabled`）。
数据集：`eval/datasets/gsm8k_style.jsonl` 10 题合成 math 题；逐题轮换执行顺序。

| 模式 | 准确率 | errors | C_API(in/out) | C_time 中位(s) |
|---|---:|---:|---:|---:|
| signal（改造后） | **9/10 = 90%** | 0 | 1388/222 | 2.53 |
| rule（改造后） | **9/10 = 90%** | 0 | 1386/235 | 2.20 |
| local_only | 9/10 = 90% | 0 | 0/0 | 1.71 |
| cloud_only | 9/10 = 90% | 0 | 233/2151 | 1.86 |

改造前同一数据集、同一环境：signal/rule 均为 7/10，3 题
`model_incomplete_answer`（本地汇总截断）。

控制组 18 题（math 10 + MMLU-style 8）在改造后为 **17/18 = 94.4%**
（math 9/10、QA 8/8），与 2026-10-25 的 local_only/cloud_only 控制组持平。

### 云端分解对照

同一 10 题、同一环境，只切换 `DOT_DECOMPOSE_TIER`：

| 分解模型 | 模式 | 准确率 | errors | C_API(in/out) | C_time 中位(s) |
|---|---|---:|---:|---:|---:|
| 本地 4B | signal | 9/10 = 90% | 0 | 1388/222 | 2.53 |
| 云端 flash | signal | 9/10 = 90% | 0 | **2643/472** | 2.74 |
| 本地 4B | rule | 9/10 = 90% | 0 | 1386/235 | 2.20 |
| 云端 flash | rule | 9/10 = 90% | 0 | **2681/319** | 2.05 |

全量 27 题：本地分解 signal/rule 81.5%，云端分解 77.8%；控制组 18 题均为
17/18。结论：云端分解在这组数据上没有准确率收益，云端 token 约翻倍，
因此默认保留本地；需要 DoT 论文式“云端分解”配置时设
`DOT_DECOMPOSE_TIER=cloud`。

## 边界

- 10 题、全部 GSM8K-style 合成题，不能外推为论文级结论。
- `math-003` 的 gold 是 9，但题意可读成 5；local_only 与 cloud_only 也答 5。
  本题是当前 10 题里的实际上限损失，不是编排造成的。
- 云端 token 输入高于 cloud_only（云端要拿到原题、子任务列表与关键步骤上下文），
  但输出 token 约为 cloud_only 的 1/10，总 token 更低。是否值得取决于端云价格口径。
- 延迟仍高于单次 local_only，因为多了依赖判断与关键步骤；这是精度换来的成本。

## 复现

```bash
# 本地模型（llama.cpp）
./llama-server.exe -m ../models/qwen3-4b-instruct-2507-q4km.gguf \
  --alias qwen-local --host 127.0.0.1 --port 8000 -c 4096 -ngl 99 --no-webui

# 评测（math 子集 + 四模式 + 逐题记录）
LOCAL_BASE_URL=http://127.0.0.1:8000/v1 LOCAL_MODEL=qwen-local \
CLOUD_API_KEY=... CLOUD_BASE_URL=https://api.deepseek.com \
CLOUD_MODEL=deepseek-flash CLOUD_THINKING=disabled \
python eval/runner.py --all --datasets gsm8k_style \
  --run-id my-math-run --records eval/my-math-run.jsonl \
  --output eval/my-math-run.md
```

逐题原始记录与运行报告保存在本机 `reports/math-dot-routing-20261122.*`
（按 `.gitignore` 的“新实验产物不入库”约定，不随 Git 分发）。
