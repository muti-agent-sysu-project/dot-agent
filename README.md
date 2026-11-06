# dot-agent · 端云协同智能体

本地小模型与云端大模型协作的本科大创项目。普通任务使用分类、拆分、依赖调度与端云路由；资料问答使用证据选择、原文回取和云端回答。多个处理角色复用模型后端，当前没有训练或合并模型权重，也没有实现通用的多轮 Agent 辩论。

**组员统一入口：本仓库的 `release/unified-team-20261025` 分支。** 当前默认 `main` 和此前的 `integration/edge-cloud-reference` 不包含本次完整的资料问答与统一页面。

## 获取代码

```powershell
git clone --branch release/unified-team-20261025 --single-branch https://github.com/Vikin-A1l/dot-agent.git
cd dot-agent
```

代码、测试、合成示例和文档随 Git 分发，无需另发代码压缩包。真实模型文件、llama.cpp 运行时、API Key、SSH 私钥和个人运行数据库不在仓库中。

## 三分钟离线体验

需要 Python 3.10 或以上版本。服务运行仅使用 Python 标准库，不需要 GPU、模型权重或 API Key。

在仓库根目录执行：

```powershell
python -m dot_agent.server --demo --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765`。点击普通任务示例观察拆分与路由；点击“资料问答示例”观察来源、引用和决策轨迹。**示例按钮会自动运行。** `--demo` 明确使用模拟后端，不读取已配置的真实模型凭据；模拟答案与指标不能作为真实性能结果。

Windows 如需独立环境，可先运行 `python -m venv .venv`，之后用 `.\.venv\Scripts\python.exe` 代替上面的 `python`，无需修改 PowerShell 执行策略。

## 文档导航

| 文档 | 内容 |
|---|---|
| [组员使用与协作指南](docs/TEAM_QUICKSTART.md) | 克隆、启动、页面使用、真实接入、更新与分支协作 |
| [代码目录与调用链](docs/CODE_MAP.md) | 从哪个文件读起、修改某项功能时看哪些模块 |
| [本次交付说明](docs/RELEASE_20261025.md) | 与既有 GitHub 版本的关系、包含内容与验证边界 |
| [统一资料问答协议](docs/UNIFIED_AGENT.md) | sources、selector、引用、预算、回退与评测 |
| [AutoDL 接入](docs/AUTODL_MAINLINE.md) | Linux 真实模型与私密配置入口 |
| [团队整合来源](docs/TEAM_INTEGRATION.md) | 各家实现的贡献与吸收范围 |
| [参考基线](docs/REFERENCE_BASELINE.md) | 独立原型及模式差异 |
| [部署纪律](docs/DEPLOYMENT.md) | 版本固定、模型校验与计量边界 |

## 系统流程

![普通任务与资料问答处理流程](docs/assets/workflow.png)

流程图为仓库内 PNG，Markdown 预览无需 Mermaid 插件或外部图床。

## 代码结构

```text
dot_agent/       当前系统：网关、分类、编排、缓存、HTTP 与网页
config/          分类规则
demo/            命令行演示
eval/            通用任务和资料路线对照；合成题集
tests/           离线回归与接口/计量测试
scripts/         AutoDL 启动等辅助工具
docs/            使用、架构、来源与验收说明
baselines/       固定来源的独立参考实现
reports/         少量保留的历史离线验收记录
```

`baselines/` 用于追溯与比较，日常业务修改从 `dot_agent/` 开始。`runtime/`、`data/metrics.db` 和新生成报告是本机运行产物，不作为代码交付内容。

## 运行检查

```powershell
python -m pip install pytest
python -m pytest tests -q -p no:cacheprovider
```

前端有改动时另做语法检查：

```powershell
node --check dot_agent/web/app.js
```

资料四路线离线流程演示：

```powershell
python eval/evidence_runner.py --demo --output reports/my-evidence-demo
```

输出目录必须尚不存在。`full/rule/bm25/local` 对照明确标记模拟，不计答案质量分。

## 真实模型接入

已配置的 AutoDL Linux 实例可运行：

```bash
python scripts/run_autodl.py server --host 127.0.0.1 --port 8765
```

前提是本地模型服务已启动、远端私密文件已配置；详见 [AutoDL 接入](docs/AUTODL_MAINLINE.md)。该脚本强制真实后端预检，启动会发送两端短生成，其中云端调用产生实际用量。不要把这条 Linux 命令用于 Windows 离线体验。

## 当前验证与研究边界

- 统一入口已有离线回归、浏览器模拟验收和合成资料对照；历史记录保留在 [离线验收](docs/UNIFIED_ACCEPTANCE_20261021.md)。
- 2026-10-25 的部署已接通 Qwen3-4B、llama.cpp CUDA 与 DeepSeek `deepseek-flash`，两端短生成和普通任务网页链路已有真实观察。
- 已观察的两项算术任务返回 6 组、48 平方米，本地调用 9 次、云端调用 0 次，流程耗时 1.746 秒。这是单题功能验收。
- 当前部署的真实资料问答、失败路径与正式效率对照尚待完成；未宣称质量、云端 token 和完整时延联合改善。
- 缓存主要是精确匹配与 TTL；天气和提醒是演示工具；当前真实接口未验证 alpha-quantile。通用评分、信号开关、阈值和完整计时仍有正式评测前需处理的边界，见 [交付说明](docs/RELEASE_20261025.md)。
