# 端云协作科研评测

冷启动衡量首次协作的效用、云成本、墙钟和暴露。持续流 ITT 从首遇开始计入每个新实例，包含技能编译、修复、失败和本地终止。EC² 官方原题、FinQA 配对题、衍生流分别报告。当前归档的是离线输入与评分工具，真实模型指标待采集。

## 数据边界与入口

- 执行层按方法规格 §5 的 Controller、Boundary、SkillRuntime 契约对接。每次网络尝试独立记账，边界绑定最终载荷、接收方、策略与许可。技能运行时只接收本地参数引用。这里的评测脚本不依赖开发中的运行时。
- FinQA 控制器逐题接收 public_inputs.jsonl 中单个 input。由可信调度器选择 split、隔离两个孪生版本的会话与文件系统。不得将整份输入、另一孪生版本或同目录挂给控制器。gt_manifest.json 仅供离线评分，manifest.json 记录源文件与输出 SHA256、拒绝清单及尝试分母。
- 公司/年度构成报告组。剔除先导组后按年份分层交错抽样，每组随机尝试一题，失败时换题，按实际尝试数计算排除率。确认和开发各自轮换三种结构。保序题比较私有优先级后选择对应账户的报表，关联题按客户名连接目录和账户文件并核对校验位，对照题可忽略敏感目录。每份选中报表及问题与公开原文逐字段比对，检查证据索引、文本位置、原程序步骤及执行答案（绝对容差 1e-5）。夹具验证选档与算术，无法证明模型实际采用了指定推理过程。
- 独立百分比增长模板生成五实例合成流，含首遇、两个新绑定、零分母和字段改名。单一模板用于机制检查。数值衍生题正式发布前仍需协议要求的人工复核。
- EC² 官方入口在工作区 科研执行/benchmarks/EC2-Bench/scripts/run.py 与 eval/run_batch.py。官方裁判读取独立工作区结果。dot_agent/collab/ec2_adapter.py 可在评测侧解析敏感项。未标注任务的 E_official 记 null，保留其他端点及失败分母。

## 离线复现

在 team-dot-agent 目录用 PowerShell 执行。生成器与审计使用标准库，测试需要 pytest、matplotlib。

~~~powershell
python -X utf8 -m eval.finqa_priv --finqa-dev "../科研方案/analysis/finqa-stage1/dev.json" --stage1-inputs "../科研方案/analysis/stage1-pilot-v2/inputs.jsonl" --output-dir "../科研执行/datasets/finqa-priv-v1" --seed 41
python -X utf8 -m pytest tests/test_finqa_priv.py tests/test_disclosure_audit.py -q
~~~

第二条命令同时验证数据重建、关联恢复、合成结果报表及出图。脱离工作区单独检出代码时，需补齐上述两个源文件才能执行全量数据测试。

## 审计与报表输入

disclosure_audit.bootstrap 接受宽表记录。每行含 stratum、family_id 或 report_group、task_id、repeat 和各臂的配对数值。先在任务内平均重复，整簇同步重采样，报告任务宏平均差。比较多臂时复用完全相同的样本行序、种子与分层。primary_gate 分别使用效用全集和已标注暴露子集。独立任务用 mcnemar，簇级 permutation 要求零假设下整簇臂标签可交换，16 簇以上使用带加一修正的 Monte Carlo。holm 按预注册端点族分别调用。interaction 接受同一 S、H 下 D、K 的 00、01、10、11 四列。

linkage 的 roster 把合成身份映射到属性字典，每个 case 含 target、family_id、background、按时序的 messages，可带 counterexample。每例针对同一接收方和别名，消息需在离线评分侧解析成属性相等约束。筛选后按均匀先验估计置信度，并列按身份字符串排序，空集视为错误。返回三条件逐例候选、恢复率、绝对及相对提升。annotation_scores 接受已仲裁标签与纳入概率，给出原始混淆矩阵及回权指标。

paper_reporting 要求预先冻结的 schedule JSONL，每行用 dataset、phase、model_pair、arm、task_id、repeat 唯一确定运行槽。results 逐槽包含相同键和 U、C_tokens、C_usd、L_s、E_official、P_ledger、attempts、unknown_usage。attempts 指全部云尝试数，unknown_usage 指 usage 未知的尝试数。本地零尝试为 0/0。失败槽保留实测指标和实际终局 U，缺失资源用 null。可提供 exposed_units、total_units 计算官方微平均。phase 取 cold、stream、ablation，后两类还需 family_id，消融行附 S、H、D、K 二值。每次导出使用同一预算、同一数据划分与来源。官方、公开孪生和隐私变体请用不同 dataset 标识。

~~~powershell
python -X utf8 -m eval.paper_reporting --schedule schedule.jsonl --results results.jsonl --output-dir tables
~~~

排期独立于结果生成。缺槽或重复槽拒绝出表。每张表的 N 为运行槽数，E_official_n 为可计算的任务数，usage、wall 列分别给出未知尝试及缺失墙钟的分子/分母。持续流的资源先按族和重复求总量，再取均值。Markdown 与 LaTeX（booktabs）并列导出。每个数据集、阶段和模型对独立绘制四个二维前沿（PNG/PDF）。阻断的成本或时延端点应先对账。图与表只描述观测值，确认性结论另用预注册门检验。

## FinQA-Priv 多臂执行器

在仓库根目录运行。输出目录须为空，避免覆盖测量账本。默认只取指定划分中的隐私孪生题，limit 按题计，每臂收到同序输入。stream 保留五个实例的原始顺序。以下命令全部在本地执行。

~~~powershell
python -X utf8 -m pytest tests -q
python -X utf8 -m eval.collab_experiment_runner --mode stub --split dev --limit 12 --output-dir "../科研执行/reports/offline-e2e-smoke-20261126"
python -X utf8 -m eval.collab_experiment_runner --mode stub --split stream --ablation all --output-dir "../科研执行/reports/offline-factorial"
~~~

也可指定 `--arms B8_modular,B9_joint`，或用 `--ablation 1011` 选择一个 S/H/D/K 组合。模拟数字只验证管线。schedule.jsonl 冻结配对，results.jsonl 保存逐槽终局分、墙钟、动作、按尝试去重的 usage、载荷摘要、保守预留和披露增量。run_manifest.json 绑定源码、数据和评分器摘要。SQLite 仅保存在本地；独立评分在全部策略结束后才读取固定哈希的 gt_manifest.json。

B1 的整题云视图仅获准用于此哈希锁定的公开财报与虚构账户，其 P 上限为 1。其余云臂的上限为 0.25，使用同一套结构化视图审查。B6 为 FinQA 的两步 PAAC 式适配，端侧绑定表、校验和受限选择替代通用 ReAct 工具环境，第二步携带有界的选择历史。B8 固定价格并单轮报价，B9 允许历史边际报价、运行期价格和经开发集估计的复用值。流上的确定性增长算子需通过替代绑定、零分母负例和后置检查才可安装；schema 漂移时放弃热命中并回到正常执行。通用运行时支持局部修复，FinQA 适配器尚未提供云生成 IR patch 的实验臂。单一流家族只能作机制验证。

audit.json 给出 B9/B8 的财报组配对 bootstrap、可用的 D×K 交互和背景／末轮／全历史三条件。冷题攻击者假设名册包含各账户的公开报表指纹；报告其恢复率、置信度及直接命中。另列八组平衡配对的合成跨轮正对照。E_FinQA_exact 是注入字段的逐轨迹直接匹配率，EC² 官方暴露列在本集保持 NA。表图在 tables/ 中，分离模拟数据。冷启动墙钟包括评测适配器建账与本地工作。stub 的墙钟只反映本机执行器开销。

### 已配置真实端点时的入口

`--mode real` 会产生真实请求。端侧配置为 COLLAB_LOCAL_BASE_URL（回环地址，含 /v1）、COLLAB_LOCAL_MODEL 和可选 COLLAB_LOCAL_API_KEY。云侧对应 COLLAB_CLOUD_BASE_URL、COLLAB_CLOUD_MODEL、COLLAB_CLOUD_RECIPIENT 和可选 COLLAB_CLOUD_API_KEY。多臂比较须提供实际隔离的接收方身份，每臂可用 COLLAB_CLOUD_<大写完整臂名>_BASE_URL 及同前缀的 MODEL、RECIPIENT、API_KEY 配置。接收方标签代表部署者核实的隔离边界，改名不能隔离同一账户。为同一部署长期复用 `--journal-dir`，保留历史披露与未知账单。适配器发送许可绑定的完整 POST 正文，不跟随重定向。预留假定输入 tokenizer 每个 UTF-8 字节至多一个 token、服务端聊天模板额外最多 512 token，部署前需核实，超界暂停该范围。

~~~powershell
python -X utf8 -m eval.collab_experiment_runner --mode real --arms B9_joint --split dev --limit 1 --profile ./development-forecast.json --journal-dir ../科研执行/recipient-journal --output-dir ../科研执行/reports/real-pilot
~~~

上述 profile 由独立开发集拟合后提供，示例路径是调用约定。JSON 顶层按 local、typed、relation、advice、fallback 分桶，各桶含 q_lower、failure_upper、remaining_tokens、remaining_seconds、future_exposure、reuse_value。缺桶时不给云动作置信承诺。可选 `--price-card` 提供 model、date 及每百万 token 的 raw_input、cache_read、output 美元单价。缺少可信分桶或单价时相应成本记 NA。正式实验仍需模型／硬件与预算冻结、独立校准、真实测量及人工审计。

## 16 题真实先导

从仓库根目录运行。启动器使用固定哈希的 FinQA-Priv v1 档案，按档案顺序取前 16 个 dev 私有实例，每次选一个臂。启动本机 llama-server，或将 AutoDL 的推理端口通过 SSH 转发到本机回环地址。租用 GPU 的时间费单列。云端使用支持非流式 chat completions 的端点，两端均须返回 prompt_tokens、completion_tokens 和明确的缓存输入桶。密钥通过相应的 API_KEY 环境变量注入。两端单次请求的输入 token 须满足上文所述字节上界。云价卡须给出该模型的实际或保守最高美元单价（涵盖计费时段）及 ISO 日期。

以下两项文件路径由运行者指定。B6_paac 可省略 profile，B8_modular 和 B9_joint 需要独立开发集冻结的五桶 profile。下列命令依次执行离线配置检查、离线上界估算、真实运行。前两条不探测网络、不创建输出目录。

~~~powershell
$pilotArgs = @('--arm','B9_joint','--profile',$profilePath,'--price-card',$priceCardPath)
python -X utf8 scripts/run_real_pilot.py @pilotArgs --preflight-only
python -X utf8 scripts/run_real_pilot.py @pilotArgs --dry-run
python -X utf8 scripts/run_real_pilot.py @pilotArgs
~~~

默认云端费用上限 2 美元，覆盖健康探针。单次完整运行最多 33 次云端、17 次本地调用（含两端各一次探针）。使用新的 --output-dir 并沿用该部署的 --journal-dir。billing.jsonl 先记预留、再记分桶回执；episodes.jsonl 逐题保存脱敏轨迹。账单未知或下一次预留会超限时中止，保留已写记录。完成后读取 run/audit.json、run/tables/cold.md、cold.tex 及 pareto-0.pdf。论文的 PilotTable 宏可指向 cold.tex；各臂真实数据就绪后再进行确认性比较。

## EC² / OpenClaw 的步级挂钩

官方 strategies.json 中的 advisory 对应 run_batch.py 的 agentic-advisor。该模式在容器内安装 cloud-advisor 插件，把完整对话发到所选云 provider 的 baseUrl。将此 baseUrl 配为容器可达、位于可信边界内的策略适配器 /v1 地址。保留真实供应商凭据在适配器一侧。官方插件自己的输入日志包含门控之前的上下文；事后统计实际云可见内容应使用网关发送记录。

每个活动轨迹创建独立 EC2StrategyPlugin，注入任务级 Registry、ViewGenerator、完整 HTTP 信封构造器、固定策略的 TransactionalBoundary 和本地 forecast 回调。预测器接收本地工具结果句柄并返回当前消息之外的预计披露；网关审查并发送实际序列化的字节。每个主体及接收方顺序执行 episode，以便将共享账本的增量归属到任务。挂钩有两种入口：

- 宿主工具循环在屏障处 await plugin.step_decide(...)，本地路径由宿主执行；工具执行后调用 record_tool_result(local_handle)，终止时调用 finalize_episode()。跨任务流将上一实例的 controller.prices 传给下一实例。K 开启时传入已验证的 skill 和本地 bindings_ref。
- 原版 advisory 插件使用 ASGI 应用 plugin.openclaw_endpoint(project, local_chat)。project(wire) 在可信侧将官方请求映射为本地状态和已声明的 schema 绑定；local_chat 必须仅连接可信端模型。云端和本地适配器均返回完整非流式 chat completion 响应字节。用 export_skill(skills_root, arm) 写入可由 setup_skills 加载的 ec2-collab/SKILL.md。新增策略不依赖修改官方六臂的 CLI choices。官方 advisory 只在调用 advisor 时经过此入口，若要在所有本地步上决策须连接宿主循环入口。

B6_paac 是别名代理委派适配；宿主负责本地 judge、工具反绑定和终止。B8_modular 使用固定价格与单消息报价。B9_joint 接受四位 shdk 覆盖，分别控制约束视图、历史报价、在线更新和技能执行。三者所有物理云请求均经过累计硬门。真实容器诊断与计费口径见下文。

## 2026-11-27 理论验证

从仓库根目录运行。独立脚本仅使用 Python 标准库，成功后在脚本旁生成 verification_summary.json。七套检查覆盖有限构造、边界及反例，蒙特卡洛用固定种子和原始单位 0.01 的绝对误差容限。标准库抽样器与原 NumPy 版本使用相同概率模型，随机轨迹会不同。

~~~sh
python -m eval.verify_theorems
pytest tests/test_theory_theorems.py -q
~~~

| 结果 | 验证内容 | 断言 |
|---|---|---:|
| 前置命题 P1/P2 | 累计闭包与并发预留反例 | 42 |
| 定理 1a / 1b | 闭包阻断高收益路径；冷编译复用的线性摊销差距 | 128 + 173 |
| 定理 2 | 随机安装、漂移、断言失败与修复的复用边界；独立模拟 | 1,413 + 15 |
| 定理 3a / 3b | 语义暴露校准界；身份关联的后验与紧界 | 283 + 806 |
| 合计 | 合成数学模型中的数值验证 | **2,860** |

## FinQA-Priv 双模型对与补充实验

U 为任务效用。费用按真实回执和对应时段价卡折算，未核供应商最终扣款，GPU 租金另计。

| 批次 | 核心结果 | 物理云请求 | 未知回执 | USD |
|---|---|---:|---:|---:|
| Pair 1，Qwen3-4B + deepseek-flash | 冷启动 B9 与 B8 的配对 ΔU 为 0，95% CI [−0.01875, 0.01875]；B9 比 B6 低 0.20。持续流 B9 v1 为 92/120 | 2,685 | 0 | 0.183022068 |
| Pair 2，Qwen3-8B + deepseek-v4-pro | 冷启动 B8、B9 v1、v2 均为 80/160；持续流分别为 120、117、119/120，v2 对 v1 的 ΔU 区间跨零。首版闭包任务 B8/v2 均未到达第二步 | 2,585 | 0 | 0.694126268 |
| Closure-Choke v2 | 24 个新工作区，B6、B8、B9 v2 的 U 为 0、0.750、0.792；B8 与 v2 均首选 typed，配对差区间跨零 | 179 | 0 | 0.012108360 |
| 三档预算 | 64 组 × 2 臂 × 3 档。B8 与 v2 的 U 在紧、中、宽档均分别为 0.4219、0.5312、0.5625。两臂各档平均 token 达标，384 槽含 2 个错误 | 128 | 0 | 0.016722332 |
| Pair 1 B9_v2 流恢复 | 原顺序重跑 24 族，120/120 正确，首次安装 24/24、热命中 48/48。存档 B8 同为 120/120，跨次时延受缓存与排队影响 | 48 | 0 | 0.000958866 |
| **合计** | 含对应探针与开发调用 | **5,625** | **0** | **0.906937894** |

## OpenBMB/EC²-Bench 真实容器诊断与续跑

冻结计划 16 题 × 6 臂，15 题至少有一个有效评分。各臂结果的分母不同，同题六臂全部评分为 0/16。实际出口由 deepseek-flash 规划，官方式（5）由 deepseek-v4-pro 审计，逻辑端侧为自托管 Qwen3-8B。

| 臂 | 有效评分/计划 | 已评分平均 U | 官方 E | 同时观测的 E − P |
|---|---:|---:|---:|---:|
| B0 local | 9/16 | 0.253 | 0 | 0 |
| B1 cloud | 14/16 | 0.759 | 0.445 | NA |
| B6 PAAC | 9/16 | 0.148 | 0.088 | NA |
| B8 modular | 8/16 | 0.176 | 0.014 | −0.486 |
| B9 joint | 7/16 | 0.209 | 0 | 0 |
| B5 advisory | 0/16 | NA | NA | NA |

47/96 为有效评分。852 次云请求含 708 笔 usage 回执、144 次 HTTP 402，未知回执 0。回执费用 $5.663854（约 $5.6639），含探针、规划与评分。拒付后停止付费队列，受影响评分从均值及配对中排除。B8 的 E − P 比较官方标注暴露与代理账本量，仅描述本配置下的经验偏差；当前样本无法验证定理 3 的总体上界或给出六臂确认性排名。

单独的离线反推实验使用本地 Qwen3-8B，无云端费用。60 条可审计轨迹中，每个条件按轨迹重复计 1,536 个目标单元，背景、末轮、全历史各命中 0、7、6。13 次命中均能在相应可见转录中逐字找到。评分只核对规范化标量，未验证实体归属，也未观察到间接推断。解析及上下文失败保留在分母。此处审计模型与端侧同家族。

仓库提供会话级策略网关、接受独立 judge 回调的实体字段审计，以及用于上述三条件实验的标量审计入口。网关运行依赖见 pyproject.toml 的 ec2 组，回归环境使用 dev 组。完整审计需要外部冻结数据、逐集原始出口和自托管 Qwen3-8B 回环端口；下列参数由运行者提供。

~~~sh
pytest tests -q
python -m eval.audit_frozen_transcripts --dataset DATASET --split FROZEN_SPLIT_JSON --episodes DIAGNOSTIC_EPISODES_CSV --runs DIAGNOSTIC_RUNS --output AUDIT_OUTPUT --port LOCAL_QWEN_PORT
~~~

1. 恢复付费服务后先核验低成本探针与回执，再恢复原诊断队列。保留旧尝试、超时、拒付及账本；新尝试分开记录，已结清请求避免重复计费。
2. 校验 ec2_split_frozen.json 的 SHA256 为 ca1476c8ae260576aa89cc6a2c659612969186bce6514d520dab7273058c8ccc。先补齐 B0/B1 同题配对（现为 8/16，缺题号 10、74、96、97、106、107、110、121），仅用诊断题拟合预测器并冻结配置；复核模板近重复。
3. 保持首批 32 个盲留任务（22 个有标注）封存至上述步骤完成，目前进度 0/32。按冻结清单在禁网容器运行，保留每次尝试、官方效用、实际出口暴露及费用。按同题配对报告估计与区间，区分无标注、无云请求、超时及拒付。
