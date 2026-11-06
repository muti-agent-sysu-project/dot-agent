# 团队主线如何接上已有 AutoDL

## 这次实际合进了什么

入口仍是 `dot_agent`。保留队友的分类、分解、依赖图调度、规则/信号路由和汇总，补入你原型中已经采用的接口与实验保护：

- 本地既支持 Ollama，也支持 `LOCAL_BASE_URL` 指定的 llama.cpp OpenAI 接口。本地调用记为 MD，不计云端 token；不会携带云端密钥。
- 云端支持 `CLOUD_THINKING=disabled`，对 DeepSeek 显式发送关闭思考的参数。
- `DOT_REQUIRE_REAL=1` 禁止缺配置时改用 stub，也关闭主线云端失败后的本地兜底。正常策略决定走本地或升级云端仍保留。
- 严格启动先各做一次短生成，检查回答完整、非空及 prompt/completion usage。只通过 `/models` 不算验证通过。
- 空回答或截断回答算失败；即使失败，已返回的有效 usage 仍进调用账本。未知 usage 不估算。接口错误不记录服务商响应正文，禁止自动跟随重定向。
- 默认演示模式仍可无模型运行。原始四模式版本在 `baselines/edge_cloud_prototype`，用于追溯和对照。

## 从哪里启动

在 **AutoDL Linux 终端**进入你实际克隆的 `dot-agent` 根目录（里面应有 `dot_agent/`、`scripts/` 和 `eval/`）。不要求重新租实例或重新下载模型。以下命令使用实例中可用的 Python；代码只依赖标准库。

先确保原来的 llama.cpp 服务已运行，监听 `127.0.0.1:8000`，模型别名为 `qwen-local`。这个整合入口不会自动启动模型或付费实例。本地模型仍沿用原部署的关闭思考设置，云端参数不会替代 llama.cpp 的启动设置。

远程密钥继续留在原位置：

`/root/autodl-tmp/edge-cloud-prototype/prototype/runtime/secrets/deepseek_api_key`

文件须为普通文件，权限600。启动脚本只在实例内读入进程环境 `CLOUD_API_KEY`，不打印内容、不放命令参数，也不复制到新项目。

```bash
# 先检查：两端分别做一次短生成，会产生少量云端费用
python scripts/run_autodl.py check

# 检查通过后，启动团队主线网页
python scripts/run_autodl.py server --host 127.0.0.1 --port 8765
```

启动网页也会重新做短生成预检。通过你已有的 SSH 连接增加本地端口转发 `8765:127.0.0.1:8765`，然后在自己电脑打开 `http://127.0.0.1:8765`。SSH 地址和端口以 AutoDL 控制台为准，不要直接把无认证的演示服务开放到公网。

脚本默认配置如下，非敏感配置可用同名环境变量覆盖：

| 配置 | 默认值 |
|---|---|
| LOCAL_BASE_URL | http://127.0.0.1:8000/v1 |
| LOCAL_MODEL | qwen-local |
| CLOUD_BASE_URL | https://api.deepseek.com |
| CLOUD_MODEL | deepseek-flash |
| CLOUD_THINKING | disabled |

脚本强制严格模式。如果本地服务未启动、密钥无效、余额不足或返回缺 usage，会报错，不能把这种运行当成功实验。

## 怎么使用、内部怎么运作

网页输入问题后，主线先检查缓存和任务类别；简单任务可直接本地回答，复杂任务分解成子任务，按依赖顺序执行。规则模式根据任务类型分配模型，信号模式再结合本地试答的一致性等信号决定是否使用云端。执行后做校验，必要时升级，最后汇总回答。面板用于看路径和指标，不代表答案已被标准答案判对。

llama.cpp 接口当前不返回本实现需要的 token 概率，因此不能宣称已验证 α-quantile 信号；现有逻辑会使用可获得的一致性等信号。

需要对照评测时，在同一目录执行下面命令。它跑主线内置题集，调用次数和费用大于短预检，运行前自行确认预算；本次整合没有代执行。

```bash
python scripts/run_autodl.py eval --all --run-id team-first-real
```

报告默认生成 `eval/report-team-first-real.md`，调用记录在 `data/metrics.db`。同名报告存在会拒绝覆盖。四组是 `local_only/cloud_only/rule/signal`，与旧原型的 `local/cloud/hybrid/dot` 不完全等价。

预检费用不计入正式题目指标；严格模式预检只打印两端模型名和输入/输出 token，不进入 SQLite。主线计量使用接口返回的 prompt+completion token，不直接保存 provider total_tokens。评测遇到未处理的失败会中止，尚未统一成失败样本报告；不要将未完成轮次当作完整对照。正式比较前还需要统一题集、评分、缓存、失败分母和重试口径。

## 本次验证边界

本次通过本地模拟 HTTP 接口验证真实客户端协议及计量行为，运行原有离线检查。没有访问远程密钥，没有启动 AutoDL，也没有调用付费云端。旧原型真实结果继续保留原来源，不作为新主线的实测成绩。

```bash
python tests/test_real_gateway.py
python tests/test_l1_rules.py
python tests/test_scheduler.py
python tests/test_tools_verify.py
python tests/test_reference_integration.py
```
