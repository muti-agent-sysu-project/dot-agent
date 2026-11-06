# 接入杨炜津的四模式基线（2026-10-03）

本次把 `Vikin-A1l/edge-cloud-prototype` 的可运行代码、组员说明、测试和真实实验记录导入主仓库，作为对照基线。主线的分类、依赖图、信号路由和Web入口继续由 `dot_agent/` 提供。此前TEAM_INTEGRATION主要吸收原则；本次补齐可以实际运行的参考实现。

## 来源与边界

- 来源仓库：https://github.com/Vikin-A1l/edge-cloud-prototype
- 固定来源提交：`22fb8c362e3ce3b811ea2265779fae58e46f7af5`。
- 导入位置：`baselines/edge_cloud_prototype/`，48个已跟踪文件按Git对象原字节导入；没有模型、密钥、runtime、虚拟环境或个人申请材料。
- 原文档中的AutoDL路径、实例和实验结果属于来源项目历史，不代表dot-agent已完成相同部署或真实实验。原项目路径仍可供有访问权限的组员使用。
- 不将历史四组成绩写成主线成绩，也不把导入顺序dot视为已经接入主线Web/DAG调度。两种算法应保留可区分的模式和实验标识。

## 从主仓库启动参考实现

Python 3.10及以上，无新增第三方依赖。在仓库根目录执行：

```bash
python eval/run_reference.py --help
python eval/run_reference.py demo --dataset data/dot_smoke.jsonl --modes local cloud hybrid dot --output reports/team-reference-demo
```

入口使用当前Python启动参考CLI；配置、题集和输出的相对路径均相对于 `baselines/edge_cloud_prototype/`。上面的结果写到该目录下的 `reports/team-reference-demo/`。目录必须不存在，再跑时换一个新名称。模拟24条请求不访问真实模型，所有结果都标记simulated，云端用量为未知。

参考模式与主线模式的对应关系：

| 参考实现 | 主线 | 解释 |
|---|---|---|
| local | local_only | 都是整题本地；公平比较仍需统一提示和生成参数 |
| cloud | cloud_only | 都是整题云端；需验证实际后端和是否发生降级 |
| hybrid | 无完全等价模式 | 参考hybrid只按整题长度/关键词分流 |
| dot | 不等同rule/signal | 参考dot是云端JSON分解、顺序执行、云端汇总；主线还有分类、建图、工具等路径 |

因此不能简单改名，把两套四模式的结果拼成同一实验。

## 真实模型使用

示例配置 `config.deepseek.example.json` 指向实例内 `127.0.0.1:8000/v1` 的qwen-local及DeepSeek官方deepseek-flash，云端关闭思考。真实运行必须在能访问对应服务的机器上进行；不会自动发现模型、读取其他项目密钥或启动付费实例。

在已部署的AutoDL内，原项目已有安全入口：

```bash
cd /root/autodl-tmp/edge-cloud-prototype/prototype
/root/miniconda3/bin/python scripts/run_autodl.py doctor
/root/miniconda3/bin/python scripts/run_autodl.py ask '17加25等于多少？只输出整数。' --mode dot
```

如果要从新克隆的团队仓库使用同一模型服务，由负责人在AutoDL当前进程内安全设置 `EDGE_CLOUD_API_KEY`（不复制到本机或Git），然后在团队仓库根目录执行：

```bash
python eval/run_reference.py --config config.deepseek.example.json doctor
python eval/run_reference.py --config config.deepseek.example.json ask '17加25等于多少？只输出整数。' --mode dot
```

这里直接使用统一环境变量入口。导入的 `scripts/run_autodl.py` 会从它自己所在项目的runtime目录找密钥，不能误认为它会读取原项目目录中的文件。迁移部署须显式配置；本次整合不复制密钥、不运行新的付费实验。

## 已有真实证据怎么用

参考实现的2026-10-02实验：6道固定合成题、4模式、24个请求。

| 模式 | 严格答对 | 云端token | 平均完整响应ms |
|---|---:|---:|---:|
| local | 1/6 | 0 | 192.22 |
| cloud | 4/6 | 391 | 658.11 |
| hybrid | 2/6 | 233 | 436.61 |
| dot | 4/6 | 2155 | 1851.73 |

历史记录证明参考实现的真实端云链路跑通过，也显示这批短题没有省token或加速收益。它不能证明主线更快、更准，亦不能证明论文效果。原始JSONL、协议哈希、评分和审计均保留在 [参考实验目录](../baselines/edge_cloud_prototype/reports/dot-verified-20261002/)。

组员入门可读 [参考实现上手指南](../baselines/edge_cloud_prototype/TEAM_GUIDE.md)，详细结果见 [老师演示说明](../baselines/edge_cloud_prototype/TEACHER_DEMO.md)。主线操作仍看根目录README。

## 验证与后续整合

2026-10-03本次Windows整合验证：主线三个测试脚本共16项通过；新增入口测试通过，实际产生24条simulated记录并验证重复输出目录拒绝覆盖；参考实现37项中35通过、2项因Linux CMake/POSIX权限条件跳过。48个来源文件逐字节一致，冻结实验文件SHA一致。未调用真实模型，未重新运行付费对照。

根目录运行主线已有检查及新增入口检查：

```bash
python tests/test_l1_rules.py
python tests/test_scheduler.py
python tests/test_tools_verify.py
python tests/test_reference_integration.py
```

运行参考实现自己的测试时，要进入其目录，以免模块和配置相对路径混淆：

```bash
cd baselines/edge_cloud_prototype
python -m unittest discover -s tests -v
python scripts/check_dot_dataset.py
```

后续应在同一模型、硬件、固定题集及评分规则下，分别运行参考dot与主线rule/signal；统一失败分母、未知usage、缓存、预热和上下文处理，再分析策略差异。本次不替换主线算法，不把历史记录冒充新主线的实测结果。

主线网关现已接入原型的 llama.cpp / DeepSeek 配置和生成校验，具体启动方法及尚未验证的部分见 [AutoDL 主线操作说明](AUTODL_MAINLINE.md)。参考目录继续保持原始提交内容。
