# AutoDL 端云协同原型

组员第一次看项目，先读 [给组员的流程说明](TEAM_GUIDE.md)：包含分工思路、AutoDL启动、Windows离线体验、四模式使用、批量评测、输出字段、代码原理和常见问题。部署配置的详细依据见本文。

按老师要求，先建立小模型与外部大模型 API 的协同及三指标测量，隐私留待后续。此版本以 AutoDL 实例为部署目标，Windows 只编写代码和测试。

“本地模型”指部署在 AutoDL GPU 实例内的小模型；“云端模型”指外部 DeepSeek API。本次使用 `config.deepseek.example.json`：`deepseek-flash`，显式关闭思考模式。默认 `config.example.json` 仍是仅本地配置；真实四模式请使用下面的安全入口。

## 已实现

- 保留 `local`、`cloud`、`hybrid`，新增 `dot`：云端 JSON 分解 → 规则分配 → 顺序执行 → 云端汇总。本地失败最多升级一次，云端失败不重试。
- 记录每次模型调用、执行路径、API token 和完整响应时间；本地 token 可在 calls 中查看，云端 token 只累计实际外部调用。
- 保留原12题，并增加6道独立核对答案的组合题。记录逐题答案、平均/P50/P95时延、成功率、准确率、云端用量和升级/退化次数。
- 环境诊断、明确标注的模拟演示、AutoDL 下载和启动脚本。Python 应用仅依赖标准库，最低 Python 3.10。

这是参考 DoT 的简化任务分解协作原型，并非论文完整复现或多智能体辩论，尚未训练新模型。规则路由不是可靠的难度分类器，本地答错不会自动升级。没有工具执行、多轮讨论、语义缓存或隐私模块。老师演示见 [TEACHER_DEMO.md](TEACHER_DEMO.md)，固定实验口径见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。

## 已租用的 AutoDL 配置

2026-10-02，用户授权后创建的实例：西北B区 **113机**，实例 ID `6fcb40b7f5-2694e599`，主机 ID `6fcb40b7f5`。单张 RTX 3080 Ti 12GB、12核 CPU、90GB RAM、30GB系统盘及50GB免费数据盘，按量创建报价 **1.14元/小时**。SSH实查GPU为NVIDIA GeForce RTX 3080 Ti，显存12288MiB，驱动580.76.05；Python3.12.3、CUDA toolkit12.8.93。

基础镜像选用 PyTorch / 2.8.0 / Python3.12 (Ubuntu22.04) / CUDA12.8。使用镜像中的CUDA toolkit在实例内编译llama.cpp。应用只依赖Python标准库，PyTorch不参与本原型模型推理。

按创建报价计算，运行2小时的算力费约2.28元、4小时约4.56元；这是单价乘时长，不是账单实付记录。运行中的实例持续计费，用完后在AutoDL控制台点击“关机”；后续开机价格及磁盘计费以平台当时显示为准。

项目实际目录：`/root/autodl-tmp/edge-cloud-prototype/prototype`。上传包内有一层`prototype/`，解压后进入这一层执行命令。

## 当前使用

Qwen3-4B-Q4_K_M后台服务已启动。2026-10-02真实验证：37/37层加载到GPU；12题全部成功返回，严格准确率6/12（50%），平均完整响应179.84ms、P50 58.83ms、P95 455.40ms。两题答错，四题附带不符合题目要求的解释。原始记录见`reports/local-verified-20261002/`；该12题合成测试只用于初版流程验收。

在AutoDL实例右侧点击“JupyterLab”，打开Terminal，执行：

```bash
cd /root/autodl-tmp/edge-cloud-prototype/prototype
/root/miniconda3/bin/python -m edge_cloud ask '17加25等于多少？只输出整数。' --mode local
```

以上12题是历史本地基线。新增四组组合题实验及真实云端生成证据见 `reports/dot-verified-20261002/`；各轮题集不同，不直接比较准确率和时延。

实例重新开机后，在一个终端运行`/root/miniconda3/bin/python scripts/start_autodl.py`并保持该终端打开，再在另一个终端运行`ask`。当前后台服务已验证在部署SSH连接关闭后仍可响应。关机结束按量实例计费，付费扩容数据盘单独计费，参见[AutoDL计费说明](https://www.autodl.com/docs/price/)。本实例使用免费50GB数据盘。

## 首次部署或在新实例复现

解压上传包到实例中你选择的项目目录，然后在该目录执行以下命令。下载、二进制、模型和运行记录均放到目录内的 `runtime/`。先保证项目所在磁盘至少10GiB可用空间。

```bash
export PATH="/root/miniconda3/bin:/usr/local/cuda/bin:$PATH"
cd /root/autodl-tmp/edge-cloud-prototype/prototype
python -m unittest discover -s tests -v
python scripts/deploy_autodl.py --preflight
python scripts/deploy_autodl.py
python scripts/start_autodl.py
```

默认部署官方 **Qwen3-4B-Q4_K_M**，文件约2.50GB；固定模型 revision 和发布方 SHA256，优先 `hf-mirror.com` 下载，失败尝试官方源，两个来源都必须通过同一校验。llama.cpp 固定 `b11321` 对应的完整commit，下载官方源码并验证本次从官方URL读取的SHA256后，在实例内编译CUDA服务。源码归档哈希是本次记录的下载值，区别于模型发布方提供的SHA256，详见 `scripts/deployment-manifest.json`。

预检查要求GPU模式、驱动主版本至少570、CUDA toolkit至少12.8、CMake至少3.18，以及g++和make。实际镜像工具未确认前不声称环境已通过。编译只写项目的 `runtime/`，配置与编译输出在 `runtime/build.log`；编译失败不会写 `deployment.json`，不会自动切换CPU。

服务绑定 `127.0.0.1:8000`，上下文4096、单并发、GPU层99；通过`chat_template_kwargs.enable_thinking=false`关闭Qwen思考模板，保留思考预算0。日志级别4记录CUDA后端和GPU模型层加载。手动执行启动脚本时在前台运行，保持终端打开。在另一个终端执行：

```bash
python -m edge_cloud doctor
python -m edge_cloud ask '17加25等于多少？只输出整数。' --mode local
python -m edge_cloud evaluate --modes local --output reports/local-first
```

`doctor` 检查配置地址下的 `/models`。默认仅本地配置会因cloud为空返回退出码2；四组配置请使用 `scripts/run_autodl.py doctor`。模型列表成功只证明认证和模型可见，仍需真实生成确认。报告目录必须是新的，拒绝覆盖。服务日志须显示 CUDA 后端和 GPU 模型层加载；如果只用CPU，不能把结果作为GPU部署验收。

若环境检查或源码编译失败，先检查缺少的构建工具以及 `runtime/build.log`，在实例项目内补齐必要工具再继续；不静默退回CPU，也不自动更新驱动。

本阶段接口地址为实例内的`http://127.0.0.1:8000/v1`。可在AutoDL的JupyterLab终端进入项目目录运行`ask`和`evaluate`。需要Windows访问时，按实例页面的实际SSH主机和端口建立SSH隧道，将本机端口转发到实例的127.0.0.1:8000。SSH密码使用终端隐藏输入。

## AutoDL 内安全运行四种模式

云端已指定为 `https://api.deepseek.com` / `deepseek-flash`。公开配置示例只存环境变量名，入口在实例内从以下私有文件读取并设置 `EDGE_CLOUD_API_KEY`：

```text
/root/autodl-tmp/edge-cloud-prototype/prototype/runtime/secrets/deepseek_api_key
```

文件权限必须为600。入口不打印密钥，不把密钥放在命令行参数中。不要把此文件下载到Windows或收入交付包。在AutoDL终端执行：

```bash
cd /root/autodl-tmp/edge-cloud-prototype/prototype
/root/miniconda3/bin/python scripts/run_autodl.py doctor
/root/miniconda3/bin/python scripts/run_autodl.py ask '买12支笔，每支8元；买5本笔记本，每本14元。总价满150元减20元，只减一次。最终应付多少元？只输出整数。' --mode dot
```

输出JSON包含最终答案、子任务分配原因、执行结果、升级/退化事件和所有调用用量。重新运行会产生云端费用。复现整轮固定实验可运行 `/root/miniconda3/bin/python scripts/run_dot_experiment.py --output reports/dot-new`，包含一次短云端生成、一次本地预热及6题×4模式；目录必须不存在，不自动重复失败实验。

示例中的隐式云端调用会产生实际费用；真实任务经规则路由可能外发原文，初版未实现隐私模块，只使用允许上传的公开/合成题。不要将申请书个人信息、老师账号或真实敏感业务材料作为题集。

## 指标口径与限制

- 完整响应时间包含规则判断、本地推理、网络与云端生成、升级等待；不以首字时间替代。
- 云端token使用API原样返回的total_tokens；输入和输出各自保留，不按字数估算。缺失usage记 `null`，已知部分为 `known_cloud_tokens_subtotal`。未外发的配置错误用量为0，但请求仍失败。dot的分解、执行、升级、汇总和退化都计入。
- 四组使用相同题集、system prompt、temperature和最终答案max_tokens。dot另有规划与中间结果开销，每题轮换模式执行顺序；服务内部缓存仍可能影响耗时。
- dot最多4个子任务，按顺序携带原问题和全部前置结果。system+user输入预算9000 UTF-8字节，不等同4096 token窗口；超限不截断，记录事件并至多退化为一次云端整题调用。原题也超限则停止；云端接口失败不重试。
- 报告的`cache_enabled: false`指应用侧不缓存答案；llama.cpp实际复用了前缀KV缓存。本轮时延来自同一热服务中的12次请求，未测模型启动耗时或Windows到实例的SSH耗时。
- 准确率用预先固定的答案完全匹配；失败计为错误。12道冒烟题只验证流程，不代表GSM8K、MMLU或真实业务表现。
- 未给出“接近”和“显著降低”的验收阈值；应与老师确认。6题单轮只能用于流程演示，不能据此宣称已经证明研究效果或统计显著的性能收益。
- 原始报告保存模型答案，使用前确认数据授权；不保存密钥、供应商错误正文或SSH配置。

## Windows离线验证

在 `prototype/` 目录运行：

```powershell
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m edge_cloud demo --output reports/demo-new
```

`demo` 的普通回复固定为42，dot规划固定为两个示例任务，所有结果标为 `simulated: true`、云端token为未知。这只检查CLI和文件输出，不属于模型实验。

设计见 `DESIGN.md`，执行计划见 `PLAN.md`，已验证和未验证证据见 `reports/verification.md`。

## 官方资料

- 模型：https://huggingface.co/Qwen/Qwen3-4B-GGUF
- 运行时：https://github.com/ggml-org/llama.cpp/releases/tag/b11321
- 固定版本服务参数：https://github.com/ggml-org/llama.cpp/blob/b11321/tools/server/README.md
- 学术网络加速：https://www.autodl.com/docs/network_turbo/
- JupyterLab终端使用：https://www.autodl.com/docs/quick_start/
