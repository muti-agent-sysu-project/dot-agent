# 隐私感知端云协同（最小披露执行）

新增模块 `dot_agent/privacy/`，把"这道题给谁做"扩展为"这道题给云端看多少"。

## 执行模式（难度 × 敏感度）

| 模式 | 条件 | 云端看到什么 |
|---|---|---|
| LOCAL_ONLY | 难度判本地 | 不参与 |
| CLOUD_PLAIN | 判云端且无敏感内容 | 原文（旧行为） |
| CLOUD_MASKED | 判云端、敏感度 LOW | 假名化文本，答案本地回填 |
| CLOUD_JUDGE | 判云端、敏感度 HIGH | 假名化问题 + 假名化本地候选，只挑选/修正 |

规则：敏感等级一旦判 HIGH，云端永远见不到原文——判别无候选或判别失败时
退化为假名化生成，不回退原文转交。

## 模块

- `sensitivity.py` 规则层敏感度：直接标识符（手机号、身份证、银行卡、邮箱、
  凭据、住址）与敏感域关键词（病历、工资、涉案等）判 HIGH；日期等准标识符判 LOW。
  `SensitivityReport.merge` 支持依赖图传播：子任务携带的前驱结论敏感则整条链从严。
- `pseudonymizer.py` 会话级实体映射：原文→假名（手机号1），跨子任务一致，
  映射表只在本地内存、随请求销毁；`leak_check` 做披露审计。
- `policy.py` `PrivacyGuard`：模式组合 + 请求级 `PrivacyLedger` 计量。
- `judge.py` 云端判别：k=3 一致性采样的候选一份数据两用（测难度 + 当候选），
  黑盒接口下任务级投机执行的落地形态。

## 集成

- `pipeline._cloud_exec`：全部云端转交（执行、校验升级、工具升级、整题收敛）
  统一过隐私判定；`DOT_PRIVACY=off` 直通，与旧版行为一致（消融对照组）。
- `signals.collect(with_samples=True)` 额外返回采样原文。
- `metrics["privacy"]`：mode_counts、entities_masked、masked/plain/judge_calls、
  leak_count——第四指标"披露"的数据来源。

## 测试

`tests/test_privacy.py`：规则、往返回填、模式矩阵、判别解析、两条端到端
（HIGH→判别不泄露、LOW→假名化回填）。e2e 过程中修掉一个真缺陷：
判别无候选时曾回退原文转交。

## 已知边界

- 人名识别暂缺（规则层无 NER）：下一步接本地模型标注或词典。
- `evidence` 工作流（带 sources 的问答路径）尚未接入隐私判定。
- 假名形如"日期1"，与原文含同形字样的极端情形会冲突（概率极低）。
