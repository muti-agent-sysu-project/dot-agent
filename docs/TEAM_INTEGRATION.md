# 团队实现整合分析（2026-10-02）

2026-10-03补充：B的完整可运行参考基线、部署脚本、组员指南及公开真实实验记录已按提交 `22fb8c3` 导入 `baselines/edge_cloud_prototype/`。下文“未吸收”指未嵌入C主流水线，相关代码现在可作为独立基线运行。来源、入口与证据边界见 [REFERENCE_BASELINE.md](REFERENCE_BASELINE.md)。

本文档记录对团队两份独立实现的详尽分析、与主线 dot-agent 的对比、以及合成决策——
每个吸收项与不吸收项都给出理由，供结题报告与后续迭代追溯。

- A = [cloud-edge-agent](https://github.com/Ultraman-Mebius-ezio/cloud-edge-agent)
- B = [edge-cloud-prototype](https://github.com/Vikin-A1l/edge-cloud-prototype)
- C = dot-agent（本仓库主线）

## 一、三份实现的定位

| 维度 | A cloud-edge-agent | B edge-cloud-prototype | C dot-agent |
|---|---|---|---|
| 核心路线 | 子任务级分配（dot）vs 整请求路由（direct）对照 | 四模式（local/cloud/hybrid/dot） | 三层分类 + 依赖图并行 + 信号路由 |
| 依赖 | litellm + rich + dotenv | **纯标准库** | **纯标准库** |
| 后端 | litellm 统一（Ollama + DeepSeek） | llama.cpp 自编译（OpenAI 兼容）+ DeepSeek | Ollama / OpenAI 兼容，桩可注入 |
| 路由 | 规则（长文本/开放词/纯算术），可解释 | 规则（长文本/关键词） | L1 规则 + L2 SLM + L3 重分类 + 信号（α-quantile/一致性/执行反馈） |
| 分解 | 云端（规划开销单独计量） | 云端 JSON（严格校验） | 本地（零云端 token），失败整题升级 |
| 调度 | 顺序执行，全量前缀 | 顺序执行，全量前缀 | 依赖图拓扑分层，同层并行，只注入直接前驱 |
| 计量 | **SQLite 双表 + 双档位 + warmup + 退避重试，最强** | 内存 + JSON 报告（unknown_usage 哲学） | 内存 meter（本次升级为 SQLite） |
| 部署 | 本机 8GB GPU | **AutoDL + 版本固定 + SHA256 双源校验，最强** | 环境变量驱动，任意后端 |
| 前端 | CLI + rich 看板 | CLI（JSON 输出） | **Web 面板（依赖图可视化）** |
| 缓存/工具/升级控制 | 无 | 无（升级一次不重试） | 语义缓存 / 工具沙箱 / 升级守卫≤K |

## 二、A（cloud-edge-agent）详评

### 长处
1. **计量工程是三份中最扎实的**，且多处是方法论级别的贡献：
   - calls/tasks 双表落库：洞察"完整响应时间是请求级量纲，落在调用级算分位数
     会把一次请求的 5 次调用算成 5 倍样本，P50/P95 直接失真"；
   - 唯一调用出口 + role/strategy 枚举登记制：调用方**无法**忘记记录，
     绕过即抛错——把口径从"约定"升级为"机制"；
   - **规划开销单独口径**（decomposer/aggregator 的 token）：指出 DoT 论文
     未计量分解与汇聚的云端开销，这正是本项目可以报告的差异点；
   - requested_tier / actual_tier 双档位：云端不可用降级本地后如实记录，
     降级率可算，不虚报云端消耗。
2. **实测工程经验密集**（注释里的踩坑记录直接可用）：
   - 冷启动首调 37.3s、预热后 0.04s——不做 warmup，真实模型 P95 必失真；
   - 瞬态错误分类 + 指数退避（RTX 5060/sm_120 冷加载偶发 CUDA 初始化失败）；
   - litellm 的 num_ctx 必须显式设（Ollama 默认 4096）、extra_body 整体替换坑、
     流式 usage 只在最后一个 chunk、DeepSeek 的 prompt_cache_hit_tokens 字段名；
   - KeyboardInterrupt 也要落库（"token 已花掉但结果被丢弃"）。
3. **本地上下文预算**：用历次调用**真实回报的 completion_tokens** 累加，
   红线"绝不用 len(text) 估"——猜测的数字会让截断时机出错且难以察觉；
   drop-oldest 单调淘汰，dropped_context 标记不会被反复推翻。
4. direct/dot 双模式**共用同一套路由规则**——对照实验只差"拆不拆"，
   不差"规则不同"，公平性设计正确。
5. 规则路由 explain() 做给定档位与规则重判的**交叉核对**，不一致显式标注。
6. 文档纪律：接口冻结文档（改接口前先读）、工作日志。

### 短处
1. 无依赖图、无并行：顺序执行 + 全量前缀 prompt 线性变长（其 README 已如实声明）；
2. 分解与汇聚走云端（消耗 token，虽已单独计量）；
3. 无语义缓存、工具沙箱、L3 重分类、Web 前端；
4. litellm 重依赖（安装链长、版本坑多——其注释里的坑大半是 litellm 的）；
5. 规则路由无难度信号（无 α-quantile / 一致性），也无升级控制；
6. 3B 模型上限（8GB 显存），与 DoT 的 8B 有差距（已如实声明）。

## 三、B（edge-cloud-prototype）详评

### 长处
1. **部署工程与可复现性是三份中最强的**：
   - llama.cpp 固定 commit（b11321）源码编译 + 本次下载 SHA256 校验；
   - 模型固定 revision + 发布方 SHA256 + hf-mirror/官方**双源下载同校验**；
   - 预检查（GPU 模式/驱动≥570/CUDA≥12.8/CMake≥3.18）；
   - **编译失败不静默回退 CPU**——"如果只用 CPU，不能把结果作为 GPU 部署验收"。
2. **诚实边界文化**值得作为团队规范：
   - 模拟输出显式标 `simulated: true`；Windows 离线 demo 与真实实验分离；
   - **usage 缺失记 null + known_cloud_tokens_subtotal**：不虚报（记 0 会高估节省）
     也不假报（len 估算会系统性偏差），未知量单独计数；
   - "模型列表成功只证明认证和模型可见，仍需真实生成确认"；
   - "6 题单轮只能用于流程演示，不能据此宣称统计显著"。
3. **JSON 严格解析**：object_pairs_hook 查重复键、键集合必须恰好 {'subtasks'}、
   逐项类型校验——LLM 输出的 JSON 校验就该这么做。
4. finish_reason==stop 才算成功；回环地址显式绕过系统代理。
5. 实验协议纪律：四模式**逐题轮换执行顺序**（防顺序偏差与前缀 KV 缓存偏爱）、
   报告目录拒绝覆盖、密钥文件 600 权限且不入交付包。
6. 真实 GPU 验证记录（37/37 层加载、12 题全返回、P50/P95 与准确率如实报告）。

### 短处
1. `system.py` 的 ask() 是约 150 行深嵌套函数，可维护性差，模式逻辑/预算/落库
   交织在一起；
2. 无依赖图并行、无难度信号、无缓存、无工具、无前端；
3. 上下文预算按 UTF-8 字节（9000B），与 4096 token 窗口不等价（已声明，但仍是近似）；
4. 云端失败不重试（对照 A 的瞬态退避重试是缺失项）；
5. 升级只一次、无 K 次上限的概念（升级事件有计数，这点好）。

## 四、合成决策（C 为主线）

原则：C 的**架构**（三层分类/依赖图并行/信号路由/工具/缓存/前端）保留，
吸收 A 的**计量机制**与 B 的**诚实性纪律**；不引入 litellm，保持纯标准库。

### 已吸收

| # | 来源 | 内容 | 落点 |
|---|---|---|---|
| 1 | A | SQLite 双表落库（calls/tasks）、run_id 实验分组 | `gateway/store.py`（新建） |
| 2 | A+B | usage 缺失记 NULL + known_subtotal + unknown_usage_calls，绝不 len 估算 | types/meter/两个客户端/汇总 |
| 3 | A | 瞬态错误分类 + 指数退避重试，失败尝试单独计量 | `gateway/metered.py` |
| 4 | A | requested/actual 双档位 + 云端降级本地兜底 + 降级率 | `gateway/metered.py` + factory 链 |
| 5 | A | warmup 预热（防冷启动污染 P95） | `gateway/warmup.py`（新建），server/eval 启动时调用 |
| 6 | A | 规划开销单独口径（DoT 未计量的差异点） | meter 汇总 `C_API_planning` + README |
| 7 | A | 真实 token 预算 + drop-oldest 上下文淘汰 | pipeline executor（字节保守近似，注释澄清与计量的区别） |
| 8 | B | finish_reason==stop 校验 | `gateway/openai_compat.py` |
| 9 | B | 回环地址绕过系统代理 | 两个真实客户端 |
| 10 | B | 逐题轮换模式执行顺序、报告拒绝覆盖、--run-id | `eval/runner.py` |
| 11 | B | 版本固定 + SHA256 双源校验的部署纪律 | `docs/DEPLOYMENT.md` |

### 未吸收（及理由）

| 内容 | 来源 | 理由 |
|---|---|---|
| litellm 统一客户端 | A | 保持纯标准库零依赖；A 注释中的 litellm 坑位已在本文档留档，需 litellm 时参考 |
| 分解/汇聚走云端 | A/B | DoT 原文分解在边缘侧；C 的本地分解零云端 token，失败再整题升级保正确性 |
| 全量前缀上下文 | A/B | C 依赖图只注入直接前驱，上下文天然受控，预算压力小 |
| direct 模式 CLI 对照 | A | C 的 eval 已有 local_only/cloud_only/rule/signal 四模式，覆盖且更全 |
| AutoDL 部署脚本 | B | 脚本与其租用实例目录结构耦合；部署纪律以 DEPLOYMENT.md 并入，脚本按需再取 |
| JSON 分解输出严格校验 | B | C 的分解是编号行格式（正则解析），不涉 JSON；若切换 JSON 分解则采用 B 的校验方式 |

## 五、遗留事项

- [ ] 真实模型下验证 warmup 与 P95 改善（A 的 37.3s→0.04s 在我们的硬件上复测）
- [ ] token 精确预算：上下文淘汰目前用字节近似，真实部署后按 usage 换算
- [ ] α-quantile 在 Ollama 各版本的 logprobs 可行性验证（决定信号路线）
- [ ] B 的 JSON 分解格式作为 decomposer 的可选模式（带严格校验）

## 主线接口接入补充（2026-10-03）

已将 llama.cpp 本地适配、DeepSeek thinking 配置、严格生成预检、截断/空回答失败且保留 usage 的处理接入 `dot_agent` 网关。新增远程密钥加载入口 `scripts/run_autodl.py`。不是只有参考代码副本；组员可直接从主线启动。操作和验证边界见 [AUTODL_MAINLINE.md](AUTODL_MAINLINE.md)。本次未做主线真实端云评测。

## 统一入口接入（2026-10-21）

在 3490928 的隔离分支上继续增量整合，新增带资料的真实主入口分支、BM25/本地编号选择、原文回取、云端引用契约和一次全文回退；网页和 JSON API 共用此主流程。普通任务框架与来源基线保留。本轮证据算法借鉴科研版 selection_products，但在线代码不导入 eval/research_data，也不更改固定 Stage 5 协议。

统一流程复用 A 的既有计量机制、B 的网关与部署纪律、C 的任务调度。另修正按 caller 分组时把 unknown usage 计入 tokens 的口径。当前只完成离线工程验证，不声明质量、速度、token 的真实联合改善。使用方式见 [统一入口](UNIFIED_AGENT.md)。
