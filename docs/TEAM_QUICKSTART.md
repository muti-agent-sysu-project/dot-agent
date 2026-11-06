# 组员使用与协作指南

本指南对应 `Vikin-A1l/dot-agent` 的 `release/unified-team-20261025` 分支。获取这个分支即可获得当前统一代码，无需向组员分发工作区压缩包。

## 1. 首次获取

在自己电脑的 PowerShell、终端或 Git Bash 中执行：

```powershell
git clone --branch release/unified-team-20261025 --single-branch https://github.com/Vikin-A1l/dot-agent.git
cd dot-agent
git branch --show-current
```

最后一行应显示 `release/unified-team-20261025`。默认 `main` 尚未包含本次完整统一功能，不能只发仓库首页让组员默认克隆。仓库若要求登录，使用组员各自拥有访问权限的 GitHub 账号。

已有旧克隆且工作区有修改时，先提交或备份自己的改动，再切分支。确认工作区干净后可以运行：

```powershell
git fetch origin
git switch --track origin/release/unified-team-20261025
```

这适用于组员自己 clone 后远端名为 `origin` 的目录。项目负责人原电脑上，团队仓库叫 `origin`、个人 fork 叫 `fork`，不要混用远端名称。

## 2. 无模型体验

需要 Python 3.10+。在仓库根目录执行：

```powershell
python -m dot_agent.server --demo --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765`。顶部后端显示 `stub` 是本节预期的模拟模式。无需 GPU 或云端 Key，按 `Ctrl+C` 结束服务。

想使用独立 Python 环境时，Windows 可运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m dot_agent.server --demo --host 127.0.0.1 --port 8765
```

无需激活脚本，也无需修改系统执行策略。Linux/macOS 使用 `.venv/bin/python`。

## 3. 页面怎么用

| 操作 | 结果与观察重点 |
|---|---|
| 只有问题，附带资料为空 | 分类、快速回答或拆任务；看 MD/MC 分工、依赖图、答案和指标 |
| 点击“资料问答示例” | 自动填充并运行 BM25 资料问答；看提供来源与答案引用 |
| 切换“本地小模型选择”后点运行 | 本地选证据编号，程序回取原文，云端据证据回答 |
| 相同问题更换资料后重跑 | 答案应根据新资料更新，资料流程绕过旧问题缓存 |
| 点击历史请求 | 回看已有结果，不重新生成；历史最多 30 条，服务重启后清空 |

点击示例会自动运行，无需再点一次运行。手动提交普通问题时，应清空之前的资料，避免继续走资料问答。资料按空行分段，页面分配 `source-1` 等编号。

可手动体验：

```text
问题：报名截止时间是什么？

资料：
活动地点在教学楼。

报名截止时间是周五17点。
```

预期答案为周五 17 点。`provided_sources` 是交给回答模型的来源，`citations` 是模型实际引用的来源；引用格式合法不等于答案事实正确。资料问答没有子任务依赖图是正常的。

## 4. 真实模型运行

两种环境要区分：`PS C:\...>` 是自己的 Windows，`root@...#` 是 SSH 后的 Linux。`scripts/run_autodl.py` 仅用于 Linux。

已有共享算力机时，组员可以用各自授权的 SSH 账号/公钥接入，在本机设置 8765 端口转发。SSH 主机、端口和授权由实例负责人提供；不共享私钥或 API Key。

**以下在算力机 Linux、仓库根目录执行：**

```bash
python scripts/run_autodl.py server --host 127.0.0.1 --port 8765
```

前提：

1. llama.cpp 已提供 `http://127.0.0.1:8000/v1`，模型别名 `qwen-local`。
2. 远端密钥文件及权限符合 [AutoDL 接入说明](AUTODL_MAINLINE.md)。
3. 云端额度、接口和模型名可用，启动短预检会产生真实云端用量。

当前默认云模型为 `deepseek-flash`，关闭思考。算力机非交互 PATH 没有 Python 时，使用 `/root/miniconda3/bin/python`。启动脚本不负责下载模型、编译 llama.cpp 或开启付费实例。

首次自建实例需要单独准备 GPU、模型权重、推理运行时和私密配置；只 clone 代码不会得到这些资源。现有模型服务无需因组员获取代码而重复部署。

## 5. 指标怎么看

- 云端 token 是云端输入加输出的已知用量；本地 token 不并入。`unknown_usage_calls` 大于 0 时不能声称总量完整。
- `local_calls` 是调用次数，不能当成 Agent 数或子任务数。
- 多个本地调用可有重叠，累计调用时间可以大于请求墙钟时间。
- 当前 `C_time_s` 在最终请求写账前取值，不包含浏览器完整往返；正式完整时延还需补齐相应测量。
- 页面不自动给答案准确率；需要人工核对或固定评分。
- 失败、重试和全文回退应保留在成本与失败分母里。
- 天气、提醒是演示实现，未接通真实天气或消息发送。

## 6. 修改与更新代码

推荐先从交付分支创建自己的功能分支：

```powershell
git switch -c feat/yourname-feature
```

修改前阅读 [代码地图](CODE_MAP.md)。保持改动集中，测试通过后提交，再推到自己有写入权限的远端并提 PR。不直接覆盖 `main`，不把密钥、权重或个人实验库提交进来。

更新共享分支时，先确认自己没有未保存的改动：

```powershell
git switch release/unified-team-20261025
git pull --ff-only
```

如果 Git 报本地有改动或不能快进，先处理自己的提交/合并关系，不使用 `reset --hard` 解决。

## 7. 验证与评测

```powershell
python -m pip install pytest
python -m pytest tests -q -p no:cacheprovider
```

只改前端时也应检查 `node --check dot_agent/web/app.js`。

资料四路线离线演示：

```powershell
python eval/evidence_runner.py --demo --output reports/my-evidence-demo
```

输出目录必须尚不存在。通用任务的 `local_only/cloud_only/rule/signal` 与资料的 `full/rule/bm25/local` 是两组不同的模式；不能把它们的 `rule` 混作同一策略。模拟模式不计质量分。

真实评测另行固定题集、预算、评分、版本、失败和缓存口径。本次代码交付不是正式科研实验；现有合成示例也不构成科研验证集。

## 8. 交付内容

| 随 Git 获取 | 各自或实例负责人另行配置 |
|---|---|
| Python 源码、网页、分类规则、测试、合成题集 | Python 环境、GPU、llama.cpp、GGUF 模型 |
| 使用说明、流程 PNG、来源与历史离线验收 | API Key、SSH 授权、供应商额度 |
| 固定来源的参考基线 | 个人运行数据库、新实验结果与日志 |

自己的完整科研数据与冻结实验产物应按独立科研协议管理；它们没有被整体混入本次协作分支。
