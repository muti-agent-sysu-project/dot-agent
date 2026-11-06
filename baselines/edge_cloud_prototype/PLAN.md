# 端云协同初版实施计划

**目标：** 同题集比较纯本地、纯云端、整请求规则协同、简化DoT任务分解协作，并记录真实token和完整响应时间。

**架构：** Python 标准库客户端对接 OpenAI 兼容服务；规则路由不调用云端；本地失败仅升级一次。所有新增内容位于 `prototype/`。

**约束：** 密钥仅用环境变量；无 usage 记未知；模拟与真实分离；不自动更改驱动；原始材料不修改。用户已批准 DESIGN.md，当前目录不是 Git 仓库，不自动创建仓库或提交。

**当前范围：** 原AutoDL实例上的小模型与DeepSeek `deepseek-flash` 非思考API；新增dot并完成6题四组联调。下面1～4记录初版已完成工作；本次进度与验收见 [IMPLEMENTATION_DOT.md](IMPLEMENTATION_DOT.md)，真实证据见 [reports/verification.md](reports/verification.md)。密钥仅在实例内安全读取，runtime不交付。

## 1. 接口、配置和执行器

- [x] 添加 `tests/test_system.py`：使用本地 HTTP 服务器断言请求与响应协议、错误处理、升级和 usage 缺失。
- [x] 运行 `python -m unittest discover -s tests -v`，确认接口尚未实现时失败。
- [x] 实现 `edge_cloud/config.py`、`client.py`、`system.py`：每次调用均记录，云端失败 token 为未知，禁重试和重定向，密钥/错误正文不入日志。
- [x] 重跑同一命令确认行为。

## 2. 评测与命令行

- [x] 添加 `edge_cloud/evaluation.py`、`__main__.py`、`data/smoke.jsonl`、`config.example.json`。
- [x] 支持 `doctor`、`ask`、`evaluate` 和明确标记的 `demo`。
- [x] 评测按题号、重复次数、模式轮换执行；汇总准确率、成功率、云端 token、平均/P50/P95 时延和升级率。
- [x] 用模拟服务验证报告，缺失 token 不计为零；空题集、重复题号和重复输出文件拒绝执行。

## 3. AutoDL 部署与真实验证

- [x] 核对官方运行时及模型元数据；模型保存发布方SHA256，运行时固定完整commit和本次官方源码下载的SHA256。因预编译包在Ubuntu24.04构建，改为在实际Ubuntu22.04镜像内编译CUDA服务。
- [x] 打包 Linux 部署脚本；在获确认的 AutoDL 实例下载到项目 `runtime/`，使用项目日志、只绑定回环地址。
- [x] 用户已确认租用；创建113机RTX3080Ti单卡实例，报价1.14元/小时；真实请求返回42，记录usage，并核对37/37层加载到GPU。
- [x] 云端配置缺失时标明未联调，不产生假的云端基线。

## 4. 交付

- [x] 添加 `README.md`、`scripts/deploy_autodl.py`、`scripts/start_autodl.py` 和 `.gitignore`。
- [x] 运行单元测试、CLI 演示和真实本地题集：Linux23项测试通过，真实12题12次请求成功、6题严格命中。
- [x] 将验证证据与限制写入 `reports/verification.md`，说明后续云端配置步骤。
