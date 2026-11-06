# 统一协作版本验收记录

日期：2026-10-21。代码基于 3490928，位于 integration/unified-evidence-20261021 隔离工作区；未提交、未推送。

## 已实现

- 普通任务继续使用原分类、快速通道、任务拆分、依赖调度和工具入口。
- Python、HTTP 和网页共用资料问答流程，支持默认 BM25 和可选本地编号选择。
- 来源编号与依赖检查、原文回取、云端回答引用检查、一次全文回退、失败计量及未知 usage 保留。
- 资料绕过问题缓存；演示结果和缓存命中的演示标记保留；--demo 不读取真实模型配置。
- 独立 full/rule/bm25/local 对照；拒绝覆盖输出，失败计入单元，模拟不计答案质量分。

## 本次实际验证

- 全套：82 passed, 7 subtests passed in 16.70s，退出码 0。
- 浏览器：BM25 资料问答、本地选择、同题更换资料、手机宽度、切回原任务拆分共 5 个场景；页面脚本错误 0。
- 四路线演示：2 条合成材料 × 4 模式 = 8 单元，流程成功 8，全部标记 simulated，答案质量评分单元为 0。
- node --check dot_agent/web/app.js 与 git diff --check 均通过。

机器记录：[验收 JSON](../reports/unified-evidence-acceptance-20261021.json)。演示记录：[四路线 summary](../reports/unified-evidence-demo-reviewed-20261021/summary.json)。本地页面截图和浏览器检查结果保存在 .validation/unified-20261021/。

## 使用

在本隔离工作区根目录运行：

```powershell
python -m dot_agent.server --demo --host 127.0.0.1 --port 8765
```

## 边界

本轮真实模型请求 0，正式 Stage 5 实验单元 0。不能根据测试或模拟数字声称省 token、加速、非劣或创新性通过。未改动原 prototype、team-dot-agent 检出和 research/phase1-p0 的科研文件；旧阶段五记录继续保持原意义。真实模型运行需要另行确认服务、上下文与资源窗口。

## 审查修复

只读审查发现并复现两处边界：零调用演示失败丢失模拟身份、账本目录异常触发额外模型调用。已增加回归测试，账本写入边界统一抛出 LedgerWriteError，阻止重试/本地兜底/全文回退；零调用和缓存命中均保留演示身份。5 项定向检查通过，最终全套结果如上；修复后只读复核通过。独立复核用例属于子集，不与全套测试数相加。
