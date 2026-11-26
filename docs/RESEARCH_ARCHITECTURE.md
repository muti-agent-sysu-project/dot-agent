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
