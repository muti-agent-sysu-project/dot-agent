# 简化 DoT 执行计划与进度

用户已批准此最小范围，2026-10-02。不重建模型、不租新实例、不增加训练、数据库或网页。

## Task 1：核心、配置、评测和测试

负责文件：edge_cloud/*.py、tests/test_dot.py（必要时局部改现有测试）、config.deepseek.example.json。其他文件由主代理处理。

保留 local/cloud/hybrid 原行为，新增 dot：云端 JSON 分解 -> 纯规则分配 -> 顺序执行（携带原问题和前置结果）-> 云端汇总。复用 ChatClient 作为唯一实际调用出口。

分解结构固定为 {"subtasks":["任务1", "任务2"]}，接受严格 JSON 对象，1～4 个非空字符串；拒绝数量越界、空、错误类型。规划提示要求只输出任务，不输出内部推理过程。有效响应但解析不合格时记录事件并退化为一次云端原题调用；实际云端 HTTP/传输/生成失败直接失败，不重试。每个本地子任务调用失败只可升级云端一次，云端再次失败即停止；保留全部调用。上下文预算以 UTF-8 bytes 明确定义，建议 9000 bytes，包含 system + 完整 user 输入，所有 dot 执行/汇总均检查；超限不截断，记录 context_budget_exceeded 并尝试一次云端原题退化（原题也超限则记录无外发失败）。该预算是工程字节限制，不宣称等同模型 token 窗口；实际溢出仍按接口失败处理。

推荐 Config 增加 dot_max_subtasks=4、dot_context_max_bytes=9000、dot_planner_max_tokens=512；普通执行/最终答案继续用 max_tokens（云端示例为512），各模式同最终限制。Endpoint 增加可选 thinking=None（只允许 enabled/disabled）；cloud 设置 disabled 时 payload 添加 thinking:{type:disabled}，不向本地强加不支持的参数。可增加 complete(..., response_format=None) 可选参数，让规划请求使用 json_object；默认参数保持老调用兼容。云端示例：https://api.deepseek.com / deepseek-flash / EDGE_CLOUD_API_KEY / 120s / thinking disabled，本地仍127.0.0.1:8000/v1 qwen-local。

每次调用必须有 request_id（整请求唯一 UUID）、call_id、stage（decompose/execute/aggregate/fallback）、subtask_id、role（local/cloud 保持旧字段兼容）、requested_model 与响应 model、success/error、finish_reason、真实usage各项、elapsed_ms。可记录公开任务的答案，不记录 reasoning_content、密钥或供应商错误正文。请求包含 subtasks（分配、原因、执行答案与成功状态）、events、upgrade_count、degradation_count、stage_metrics、error。请求总云端token与stage_metrics必须由同一 calls 列表聚合，不重复计数；缺失usage为null并记录known subtotal；纯配置错误未外发=0。本地token可单独汇总；保留原calls字段以兼容旧报告。

evaluate 新增 dot 支持；请求与各call关联 experiment_id/task_id/repeat（可在收到row后补充，不更改ask已有位置参数）。汇总包含升级和退化次数、各阶段云端用量/调用耗时、准确率/成功率/mean/P50/P95。模拟模式所有请求和汇总显式simulated，云端量为未知；模拟 dot 应生成有效子任务和固定示例结果来展示真实编排，但不能读取题集答案或冒充模型能力。旧模式仍保持固定模拟答案42。

按TDD增加真实本地HTTP测试，先观察新增功能失败再实现：正常计划/顺序/前置信息/混合分配、非法和1/4/5任务边界、分解云端失败、local失败只升级一次、云端失败不重试、预算超限退化及原题超限、每阶段未知usage与总量、显式禁思考payload、请求标识关联、原有三模式回归、模拟标记。阅读适用 TDD 技能。不要调用真实模型或接触runtime/secrets，不安装依赖，不创建Git仓库/提交。当前非Git工作区；用原文件备份生成审查diff由主代理处理。

从 prototype/ 运行 .venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -v，及 compileall。完整实现报告写 runtime/dot-core-report.md，包含RED/GREEN命令和结果、改动与疑虑；简短返回状态和路径。

## Task 2：联通、题集、部署和实验（主代理）

- [x] 读取实例页面状态；只访问原实例。密钥仅远端runtime/secrets/deepseek_api_key读取。
- [x] 新增6题组合题集与独立答案计算校验，不修改smoke题集；运行前固定配置与SHA。
- [x] 短cloud请求验证完整usage，local预热；24请求全部返回。额外独立演示一次，返回146；包含预检与演示共24次云端调用/3252tokens，未重跑评测。
- [x] 远端runtime/before-dot-20261002备份原代码并同步；Linux37项测试全部通过，实验报告新目录保留。

## Task 3：审查和交付

- [x] 审查核心代码与测试，修复题集哈希预检、原始分配字段、退化比例及usage预检；37项测试完成。请求阶段补充本地计量与全部调用耗时。
- [x] 中文老师演示说明：流程图、真实例子、四组指标、局限与下一步。
- [x] 更新README/DESIGN/PLAN/verification，保留历史证据且清除当前状态矛盾。
- [x] 46文件白名单交付ZIP及SHA，排除runtime/虚拟环境/密钥/个人材料；压缩CRC、文件内容及敏感特征检查通过，远端同步并校验。
- [x] 逐条完成审计，最终独立审查无实质需修项。原实例仍运行且按量计费，health=ok；创建报价1.14元/小时。密钥权限600，未输出或复制。验收记录见reports/verification.md。

## 最终验收映射

1. 代码：在原edge_cloud内局部新增dot，保留三模式；无新框架、训练、数据库或网页。
2. 指标：reports/dot-verified-20261002/audit.json验证24请求、50调用、ID、评分、阶段与总量及延迟；云端用量均真实且已知。
3. 实验：6题单轮4组，预检与演示单列，无事后改评分或挑选结果；dot没有省token或加速的证据。
4. 演示：README和TEACHER_DEMO内完整命令已在AutoDL实际运行，返回146；原始输出demo-ask.json。
5. 测试：Linux37全通过，Windows35通过/2平台跳过，compileall通过；两次独立审查已收口。
6. 交付：README、DESIGN、PLAN、verification与老师说明一致；ZIP及sidecar在本地prototype/和远端项目上一级，源码及文档同步原目录。
