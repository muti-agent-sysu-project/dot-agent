# 统一端云证据问答实施计划

> 使用 executing-plans 在本会话逐项执行；用户已确认方案 1，无须重复申请设计确认。

**目标：** 在团队实际主入口接通带来源的证据问答，保留普通任务框架、实际计量和独立对照。
**架构：** 在干净主线 3490928 的隔离工作区增量实现。证据模块不依赖科研 eval；所有模型请求复用 MeteredClient。
**技术栈：** Python 3.10+、标准库；现有 pytest 作开发检查，不新增依赖。

## 约束

- 不修改 prototype、team-dot-agent 原检出或 research/phase1-p0 的现有改动。
- sources 缺省时保持旧流程；资料请求绕过仅按问题匹配的缓存。
- API 支持 bm25/local；对照入口另外运行 full/rule，均使用相同回答契约。
- 提供证据与答案引用分开记录。契约检查不声称事实正确。
- 最多一次流程级全文回退；失败、重试、未知 usage、墙钟时间全部保留。
- 不启动远端、不使用现有凭据发收费请求、不改固定 Stage 5 协议。

## Task 1：输入与证据选择

文件：新增 dot_agent/orchestrator/evidence.py、tests/test_evidence_pipeline.py；修改 dot_agent/types.py。
- [x] 先写并运行失败测试：重复来源、布尔/越界/重复索引、原文保持、BM25、引用范围。
- [x] 实现与研究代码同原则的通用选择与严格解析；保留表格头部依赖的显式来源映射测试。
- [x] 专项检查通过。

## Task 2：接入主流程与计量

文件：修改 dot_agent/pipeline.py；扩展同一测试文件。
- [x] 先证明主入口尚不接受 sources。
- [x] 接入 BM25/local 两种选择，统一云端回答契约、一次回退和清晰失败结果。
- [x] 验证失败回退的调用数、未知 usage、完整时间、tasks/calls 落库和缓存隔离。
- [x] 资料回答禁用云端失败后的静默本地兜底；保留普通请求旧行为。

## Task 3：服务入口与可用演示

文件：修改 dot_agent/server.py、gateway/demo_behavior.py；新增 tests/test_evidence_server.py；必要时局部修改 web/app.js、web/index.html。
- [x] 先写 HTTP 测试，再接入 JSON 资料输入和结果序列化。
- [x] 保持原 query 请求兼容；输入错误在调用前返回 400；失败结果携带计量与安全错误码。
- [x] 演示模型返回明确模拟的结构化输出，网页提供可选资料输入并展示引用。

## Task 4：独立资料对照与交付

文件：新增 eval/evidence_runner.py、eval/datasets/evidence_smoke.jsonl、tests/test_evidence_eval.py；更新 README.md、docs/TEAM_INTEGRATION.md。
- [x] 四路线全文/固定规则/BM25/小模型采用同一契约与计量；输出拒绝覆盖，模拟与真实分开。
- [x] 全套 pytest、离线四路线 smoke、浏览器/API 检查和 git diff --check。
- [x] 更新本计划实测证据与文档；保留未运行真实模型和科研效果的边界。

## 执行证据

- 全套：82 passed, 7 subtests passed in 16.70s，exit 0。
- 浏览器：5 个场景通过，页面脚本错误 0。
- 四路线 smoke：8 单元成功，simulated，不评分。
- 两个审查问题先复现后修复，5 项定向检查通过，修复后复核通过。
- JS 语法及 git diff --check 通过；未运行真实模型、未启动 Stage 5 正式实验、未提交推送。
- 详见 docs/UNIFIED_ACCEPTANCE_20261021.md 及 reports/unified-evidence-acceptance-20261021.json。
