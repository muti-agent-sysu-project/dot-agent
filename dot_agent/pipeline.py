"""端到端编排流水线（框架核心）。

流程（对应方案架构图）：
  缓存查询 → 分类（L1/L2）→ 快速通道或重流水线 → 汇总 → 写缓存 → 记指标

快速通道：策略 decompose=False（QA）→ 本地直答 + k=3 一致性校验。
重流水线：分解 → 依赖图 → 逐层并行执行（每子任务：信号收集 → 路由分配 →
生成 → 校验 → 失败升级，受 EscalationGuard 约束）→ 本地汇总。

L3 运行时重分类钩子：快速通道一致性不过 → 升级 REASONING 走完整流水线；
分解失败 → 整题收敛云端。全部事件写入 trace（"失败升级"统计来源）。

计量：local/cloud 均以 MeteredClient 包装，覆盖全部 caller 路径。
"""
from __future__ import annotations

import os
import sys
import time

from .cache.semantic import SemanticCache
from .classifier.l1_rules import extract_features
from .classifier.l2_semantic import SemanticClassifier
from .classifier.pipeline import ClassificationPipeline
from .gateway.base import ModelClient
from .gateway.meter import CallMeter
from .gateway.metered import MeteredClient
from .orchestrator.decomposer import TaskDecomposer
from .orchestrator.escalator import EscalationGuard
from .orchestrator.evidence import EvidenceWorkflow, validate_sources
from .orchestrator.router import POLICY, Router
from .orchestrator.scheduler import TaskScheduler
from .orchestrator.signals import SignalCollector
from .orchestrator.tools import default_registry
from .orchestrator.verify import Verifier
from .privacy import ExecMode, PrivacyGuard, cloud_judge
from .types import Caller, Label, ModelTier, RunResult, SubTask


def _budget(key: str, default: int) -> int:
    """预算只是上限，写多少用多少；可用环境变量按实测再调。"""
    try:
        return max(1, int(os.environ.get(key, default)))
    except ValueError:
        return default


BUDGET_LOCAL_ANSWER = _budget("DOT_BUDGET_LOCAL_ANSWER", 768)
BUDGET_CLOUD_ANSWER = _budget("DOT_BUDGET_CLOUD_ANSWER", 1024)
BUDGET_TOOL_CLOUD = _budget("DOT_BUDGET_TOOL_CLOUD", 512)
BUDGET_SUMMARY = _budget("DOT_BUDGET_SUMMARY", 1536)


class AgentPipeline:
    def __init__(
        self,
        local_client: ModelClient,
        cloud_client: ModelClient,
        meter: CallMeter | None = None,
        theta_default: float = 0.5,  # 兼容旧签名；θ 现由 POLICY/标定文件决定
        theta_overrides: dict | None = None,  # 显式覆盖标定值（消融用）
        use_signals: bool = True,  # False → v0 规则协同（信号关闭，仅 bias 路由）
        run_id: str = "default",   # 实验分组（SQLite 落库）
        mode: str = "auto",        # auto/rule/signal/local_only/cloud_only
        context_budget_bytes: int = 8192,  # 本地子任务上下文预算（保守近似）
    ) -> None:
        self.use_signals = use_signals
        self.run_id = run_id
        self.mode = mode
        self.context_budget_bytes = context_budget_bytes
        self.meter = meter or CallMeter()
        # 计量包装：所有组件只持有 MeteredClient，调用即记录（内存+SQLite）。
        # 云端挂本地兜底：云端不可用时降级执行并如实记双档位（降级率可算）
        self.local: MeteredClient = MeteredClient(local_client, self.meter, run_id=run_id)
        self.cloud: MeteredClient = MeteredClient(
            cloud_client, self.meter, run_id=run_id, fallback=None if os.environ.get("DOT_REQUIRE_REAL") == "1" else self.local
        )
        self.classifier = ClassificationPipeline(SemanticClassifier(self.local))
        self.decomposer = TaskDecomposer(self.local)
        self.scheduler = TaskScheduler(self.local)
        self.router = Router(theta_overrides)
        # θ 自适应：DOT_ADAPTIVE=on 时用本地失败样本流在线微调阈值
        from .adaptive import ThetaAdapter
        self.theta_adapter = ThetaAdapter(
            {label: theta for label, theta in self.router._theta.items()},
            enabled=os.environ.get("DOT_ADAPTIVE", "off") == "on")
        self.signals = SignalCollector(self.local)
        self.verifier = Verifier()
        self.tools = default_registry()
        self.cache = SemanticCache()
        # 隐私模块每次请求重建（独立实体映射表）；DOT_PRIVACY=off 直通作消融对照
        # NER 后端按 DOT_NER_BACKEND 选择：rules（默认）/ transformers / presidio / off
        self.privacy_enabled = os.environ.get("DOT_PRIVACY", "on") != "off"
        self._ner_backend = None
        if self.privacy_enabled and os.environ.get("DOT_NER_BACKEND", "rules") != "off":
            from .privacy.ner import get_backend
            self._ner_backend = get_backend()
        self._privacy = PrivacyGuard(enabled=self.privacy_enabled, ner_backend=self._ner_backend)

    # ------------------------------------------------------------------ #
    async def run(self, query: str, *, sources: list[dict] | None = None,
                  selector: str = "bm25") -> RunResult:
        """对外请求入口：模型侧异常收敛成失败结果，不让 HTTP 层返回裸 500。

        校验类 ValueError 仍向上抛出（HTTP 层按 400 处理）；模型调用失败
        （截断/空答/连接/HTTP 错误）返回 success=False 且带错误码的结果，
        调用账本里已产生的真实用量照常保留。
        """
        t0 = time.perf_counter()
        self._privacy = PrivacyGuard(enabled=self.privacy_enabled,
                                     ner_backend=self._ner_backend)
        try:
            return await self._run(query, sources=sources, selector=selector)
        except RuntimeError as error:  # GenerationError / model_http_* / model_connection_error
            code = str(error) or type(error).__name__
            print(f"[pipeline] 请求失败: {code}", file=sys.stderr)
            trace = [
                f"模型调用失败: {code}",
                "本地侧失败会先尝试升级云端；仍失败则如实返回失败，不伪造答案",
            ]
            return self._finalize(query if isinstance(query, str) else "", "",
                                  Label.QA, "failed", trace, t0,
                                  success=False, error=code)

    async def _run(self, query: str, *, sources: list[dict] | None = None,
                   selector: str = "bm25") -> RunResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("invalid_query")
        if selector not in ("bm25", "local", "full", "rule"):
            raise ValueError("invalid_selector")
        t0 = time.perf_counter()
        if sources is not None:
            sources = validate_sources(sources)
            cloud = MeteredClient(self.cloud.inner, self.meter, run_id=self.run_id,
                                  retries=self.cloud.retries)
            outcome = await EvidenceWorkflow(self.local, cloud, self.meter,
                                             privacy=self._privacy).run(query, sources, selector)
            return self._finalize(
                query, outcome.answer, Label.QA, outcome.path, outcome.trace, t0,
                success=outcome.error is None, error=outcome.error,
                provided_sources=outcome.provided_sources, citations=outcome.citations)
        if selector != "bm25":
            raise ValueError("selector_requires_sources")
        trace: list[str] = []
        meter = self.meter

        # 1. 语义缓存（仅确定性域可命中）
        features = extract_features(query, self.classifier.rules)
        cached = self.cache.get(query, features.deterministic)
        if cached is not None:
            trace.append(f"cache_hit（deterministic={features.deterministic}）")
            return self._finalize(query, cached, Label.QA, "cache_hit", trace, t0)

        # 2. 分类：L1 规则层短路（零推理成本）；不确定时分类与分解合并推理
        #    （旧路径 L2 分类 + 分解是两段串行本地推理，合并后省一整段时延）
        cls = self.classifier.classify_l1_stage(query)
        merged_subtasks: list[SubTask] | None = None
        if cls is not None:
            meter.stage("classify")
            meter.stage_end("classify")  # 纯规则，耗时≈0，保留阶段名兼容面板
            trace.append(
                f"classify: {cls.label.value} conf={cls.conf:.2f} src={cls.source} "
                f"fast_path={cls.fast_path}（约束={features.constraint_markers} 数学式="
                f"{features.has_math_expr} 选项={features.has_options} 工具词={features.action_verbs}）"
            )
        else:
            from .types import Classification  # 局部导入避免循环依赖

            meter.stage("decompose")
            merged_label, merged_subtasks = await self.decomposer.decompose_and_classify(query)
            meter.stage_end("decompose")
            cls = Classification(merged_label or Label.REASONING, 0.60,
                                 "merged", features, fast_path=False)
            trace.append(
                f"分类+分解合并推理: label={cls.label.value} src=merged"
                + ("" if merged_subtasks else "（分解失败）")
            )
            if merged_subtasks is None:
                guard = EscalationGuard()
                guard.converge_to_cloud()
                trace.append("合并分解失败 → 整题收敛云端（direct_cloud）")
                answer = await self._cloud_exec(
                    query, caller=Caller.EXECUTE, budget=BUDGET_CLOUD_ANSWER,
                    privacy_texts=[query],
                )
                return self._finalize(query, answer, cls.label, "direct_cloud", trace, t0)
        policy = POLICY[cls.label]

        # 3. 快速通道：decompose=False（QA 类）→ 本地直答 + 一致性校验
        if cls.fast_path or policy.decompose is False:
            answer, ok = await self._fast_path(query, policy.verify, trace)
            if ok:
                self.cache.put(query, answer, features.deterministic)
                return self._finalize(query, answer, cls.label, "fast_path", trace, t0)
            # L3 重分类：QA 一致性不过 → 升级 REASONING 走完整流水线
            trace.append("L3 重分类: QA → REASONING（一致性校验未通过）")
            from .types import Classification  # 局部导入避免与模块头部循环依赖

            cls = Classification(Label.REASONING, 0.5, "reclassify", features, False)
            policy = POLICY[cls.label]

        # 4. 重流水线（REASONING / PLANNING / TOOLUSE.dynamic）
        guard = EscalationGuard()
        if merged_subtasks is not None:
            subtasks = merged_subtasks  # 合并推理已产出，跳过单独分解
        else:
            meter.stage("decompose")
            subtasks = await self.decomposer.decompose(query)
            meter.stage_end("decompose")

        if subtasks is None:
            guard.converge_to_cloud()
            trace.append("分解失败 → 整题收敛云端（direct_cloud）")
            answer = await self._cloud_exec(
                query, caller=Caller.EXECUTE, budget=BUDGET_CLOUD_ANSWER,
                privacy_texts=[query],
            )
            return self._finalize(query, answer, cls.label, "direct_cloud", trace, t0)

        trace.append(
            f"分解出 {len(subtasks)} 个子任务: "
            + "; ".join(f"{st.id}:{st.description[:12]}" for st in subtasks)
        )

        meter.stage("execute")
        answers = await self.scheduler.run(
            query, subtasks, self._make_executor(cls, policy, guard, trace)
        )
        meter.stage_end("execute")

        # 5. 本地汇总（零云端成本）；截断时按需升级云端一次（优化 5）
        meter.stage("summarize")
        summary_prompt = (
            f"原任务: {query}\n各子问题及答案:\n"
            + "\n".join(f"{st.id}: {st.description} -> {answers[st.id]}" for st in subtasks)
            + "\n请给出最终答案。"
        )
        final_text = None
        try:
            final = await self.local.generate(
                summary_prompt, max_tokens=BUDGET_SUMMARY, temperature=0.0,
                caller=Caller.SUMMARIZE
            )
            final_text = final.text
            trace.append("本地汇总完成")
        except RuntimeError:
            # 汇总被截断/空答：拼接质量差，按需升级云端汇总一次
            # （Guard 额度留给子任务执行，这里用独立的一次性允许）
            try:
                trace.append("本地汇总失败（截断/空答）→ 升级云端汇总一次")
                final_text = await self._cloud_exec(
                    summary_prompt, caller=Caller.SUMMARIZE,
                    budget=BUDGET_SUMMARY, privacy_texts=[summary_prompt],
                )
            except RuntimeError:
                final_text = None
        if final_text is None:
            # 云端汇总也不可用：退化为按顺序拼接子任务答案，不作废整次请求
            final_text = "\n".join(f"{st.id}: {answers[st.id]}" for st in subtasks)
            trace.append("云端汇总不可用 → 拼接子任务答案兜底")
        meter.stage_end("summarize")
        self.cache.put(query, final_text, features.deterministic)
        views = {v["id"]: v for v in self._privacy.ledger.views}
        subtask_data = [
            {
                "id": st.id,
                "description": st.description,
                "deps": list(st.deps),
                "assigned": st.assigned.value if st.assigned else None,
                "level": st.level,
                "status": st.status,
                # 面板要展示完整中间过程，保留到 2000 字，够长任务用
                "answer": st.answer[:2000],
                # "云端所见"：该子任务转交云端时的实际文本（截断），直观展示假名化
                "cloud_view": views.get(st.id),
            }
            for st in subtasks
        ]
        return self._finalize(query, final_text, cls.label, "pipeline", trace, t0, subtask_data)

    # ------------------------------------------------------------------ #
    def reset_meter(self) -> None:
        """重置计量器（服务端/评测逐请求计量用；缓存等状态保留）。"""
        self.meter = CallMeter()
        self.local.meter = self.meter
        self.cloud.meter = self.meter

    async def _fast_path(self, query: str, verify_kind: str, trace: list[str]) -> tuple[str, bool]:
        """本地直答；k=3 一致性（复用 SignalCollector 的 greedy 结果当答案）。"""
        sig, greedy = await self.signals.collect(query)
        if greedy is None:  # 本地试答被截断/空答/连接失败 → 交给上层升级
            trace.append("fast_path 本地试答不可用（截断/空答/失败）→ 升级完整流水线")
            return "", False
        trace.append(
            f"fast_path 信号: α-quantile={sig.alpha_quantile} "
            f"consistency={sig.consistency} → 本地直答"
        )
        ok = sig.consistency == 1.0 and self.verifier.verify(verify_kind, query, greedy.text)
        return greedy.text, ok

    # 思考型云端模型（GLM-5.3-flash 等）reasoning 也计入 completion，
    # 长答案（写代码/长推理）容易被 budget 截断。截断时预算翻倍重试一次，
    # 上限 DOT_BUDGET_CLOUD_MAX（默认 8192），仍截断则返回部分答案并注明。
    CLOUD_BUDGET_MAX = _budget("DOT_BUDGET_CLOUD_MAX", 8192)

    async def _cloud_generate(self, prompt: str, *, budget: int, caller: Caller):
        """带截断重试的云端生成：incomplete → 预算翻倍一次 → 仍不行返回部分文本。

        思考模型 reasoning 吃掉预算导致 finish_reason=length 时，直接把整次
        请求判失败太浪费（36s+ 的调用全作废）；这里先用翻倍预算重试，仍截断
        则把部分答案交给上层（由 trace 注明），不再抛异常作废请求。
        """
        from .gateway.openai_compat import GenerationError

        try:
            return await self.cloud.generate(prompt, max_tokens=budget,
                                             temperature=0.0, caller=caller)
        except GenerationError as error:
            if str(error) != "model_incomplete_answer":
                raise
            retry_budget = min(budget * 2, self.CLOUD_BUDGET_MAX)
            if retry_budget <= budget:
                raise  # 已到上限，无法再扩
            print(f"[pipeline] 云端答案截断，预算 {budget}->{retry_budget} 重试",
                  file=sys.stderr)
            return await self.cloud.generate(prompt, max_tokens=retry_budget,
                                             temperature=0.0, caller=caller)

    async def _cloud_exec(
        self, prompt: str, *, caller: Caller, budget: int,
        privacy_texts: list[str] | None = None,
        candidates: list[str] | None = None,
        view_label: str = "",
    ) -> str:
        """隐私感知的云端转交：按敏感度决定云端以什么身份参与。

        CLOUD_PLAIN   原文转交（无敏感内容，旧行为）
        CLOUD_MASKED  假名化转交，答案本地回填
        CLOUD_JUDGE   云端只在（假名化后的）本地候选中挑选/修正；
                      判别调用失败时退化为假名化生成，不退回原文转交
        view_label: 子任务 id，用于把实际发出的文本记进面板"云端所见"。
        """
        privacy = self._privacy
        mode, report = privacy.classify(
            True, privacy_texts if privacy_texts is not None else [prompt])

        def record_view(text: str) -> None:
            if view_label:
                privacy.ledger.views.append(
                    {"id": view_label, "mode": mode.value, "text": text[:500]})

        if mode is ExecMode.CLOUD_JUDGE:
            if not candidates:
                # 无候选（采样不可用/信号关闭）：判别无从谈起，退化为假名化生成，
                # 绝不回退原文转交——敏感等级一旦判 HIGH，云端就不见原文
                mode = ExecMode.CLOUD_MASKED
                privacy.ledger.record(mode)
            else:
                masked_q = privacy.pseudonymize(prompt, report)
                masked_cands = [privacy.pseudonymize(c) for c in candidates]
                record_view(masked_q)
                try:
                    picked, ok = await cloud_judge(self.cloud, masked_q, masked_cands)
                except RuntimeError:  # 云端不可用/降级也失败：走假名化生成兜底
                    picked, ok = "", False
                if ok:
                    privacy.ledger.judge_calls += 1
                    # 判别输出可能是"修正"而非某条候选的复述，与 candidates[0]
                    # 做实体数守恒对比会误报；original="" 使溢出校验跳过，
                    # 假名残留检查（回填遗漏）仍然生效
                    return privacy.restore(picked, original="")
                mode = ExecMode.CLOUD_MASKED
                privacy.ledger.record(mode)
        if mode is ExecMode.CLOUD_MASKED:
            privacy.ledger.masked_cloud_calls += 1
            masked = privacy.pseudonymize(prompt, report)
            record_view(masked)
            out = (await self._cloud_generate(
                masked, budget=budget, caller=caller
            )).text
            return privacy.restore(out, original=prompt)
        privacy.ledger.plain_cloud_calls += 1
        record_view(prompt)
        return (await self._cloud_generate(
            prompt, budget=budget, caller=caller
        )).text

    def _make_executor(self, cls, policy, guard: EscalationGuard, trace: list[str]):
        label, verify_kind = cls.label, policy.verify

        async def executor(st: SubTask, ctx: dict[str, str]) -> str:
            # TOOLUSE：执行即分类器——先本地沙箱执行，错误码即难度信号
            if label is Label.TOOLUSE:
                tr = self.tools.try_execute(st.description)
                if tr is not None:
                    if tr.ok:
                        st.assigned = ModelTier.MD
                        trace.append(f"{st.id} 工具执行成功（本地沙箱）")
                        return tr.output
                    trace.append(f"{st.id} 工具执行失败[{tr.error_code}]")
                    if guard.can_escalate():
                        guard.escalate()
                        st.assigned = ModelTier.MC
                        trace.append(
                            f"{st.id} → 升级云端（guard 剩余额度 {guard.remaining}）"
                        )
                        return await self._cloud_exec(
                            st.description, caller=Caller.RETRY,
                            budget=BUDGET_TOOL_CLOUD, privacy_texts=[st.description],
                            view_label=st.id,
                        )
                    trace.append(f"{st.id} 升级额度耗尽 → 保留失败输出")
                    return tr.output

            # 上下文裁剪：只注入直接前驱答案；超预算丢弃最早前驱（drop-oldest，
            # 单调淘汰）。字节预算是保守近似——这是上下文装配策略而非计量，
            # 计量仍只用后端真实 usage（吸收自团队 cloud-edge-agent 的预算思想）
            kept = [d for d in st.deps if d in ctx]
            total_b = sum(len(ctx[d].encode("utf-8")) for d in kept)
            headroom = self.context_budget_bytes - len(st.description.encode("utf-8"))
            while kept and total_b > headroom:
                dropped_dep = kept.pop(0)
                total_b -= len(ctx[dropped_dep].encode("utf-8"))
                trace.append(f"{st.id} 上下文预算超限，丢弃最早前驱 {dropped_dep}")
            ctx_part = "".join(f"\n已知 {d}: {ctx[d]}" for d in kept)
            prompt = f"子问题: {st.description}{ctx_part}\n请给出简洁答案。"
            # 敏感度判定的输入含前驱结论（依赖图传播：上文敏感则整条链从严）
            p_texts = [st.description] + [ctx[d] for d in kept]

            if self.use_signals:
                sig, greedy, sample_texts = await self.signals.collect(prompt, with_samples=True)
            else:  # v0 规则协同：无难度信号，路由仅按类型默认倾向
                sig, greedy, sample_texts = None, None, []
            decision = self.router.assign(label, sig)
            self.theta_adapter.apply_to(self.router)  # 自适应开时同步最新 θ
            st.assigned = decision.tier
            trace.append(f"{st.id} 路由: {decision.reason}")

            if self.use_signals and greedy is None:
                # 本地采样拿不到有效样本：直接交给云端，不再用同一模型重试同样的失败
                st.assigned = ModelTier.MC
                trace.append(f"{st.id} 本地采样不可用 → 云端执行")
                return await self._cloud_exec(
                    prompt, caller=Caller.EXECUTE, budget=BUDGET_CLOUD_ANSWER,
                    privacy_texts=p_texts, view_label=st.id,
                )

            if decision.tier is ModelTier.MD:
                if greedy is None:  # 信号关闭时本地直接生成
                    greedy = await self.local.generate(
                        prompt, max_tokens=BUDGET_LOCAL_ANSWER, temperature=0.0,
                        caller=Caller.EXECUTE
                    )
                answer = greedy.text  # 复用 greedy 试答，省一次生成
                ok = self.verifier.verify(verify_kind, st.description, answer)
                # 自适应观测：本地校验结果反馈给 θ 调节
                self.theta_adapter.observe(label, "local_ok" if ok else "local_fail")
            else:
                answer = await self._cloud_exec(
                    prompt, caller=Caller.EXECUTE, budget=BUDGET_CLOUD_ANSWER,
                    privacy_texts=p_texts, candidates=sample_texts or None,
                    view_label=st.id,
                )
                ok = True

            if not ok and decision.tier is ModelTier.MD:
                if guard.can_escalate():
                    guard.escalate()
                    st.assigned = ModelTier.MC
                    trace.append(f"{st.id} 校验不过 → 升级云端（guard 剩余额度 {guard.remaining}）")
                    answer = await self._cloud_exec(
                        prompt, caller=Caller.RETRY, budget=BUDGET_CLOUD_ANSWER,
                        privacy_texts=p_texts, candidates=sample_texts or None,
                        view_label=st.id,
                    )
                else:
                    trace.append(f"{st.id} 校验不过且升级额度耗尽 → 保留本地答案")
            return answer

        return executor

    # ------------------------------------------------------------------ #
    def _finalize(self, query, answer, label, path, trace, t0, subtasks=None, *,
                  success=True, error=None, provided_sources=None, citations=None) -> RunResult:
        total = time.perf_counter() - t0
        m = self.meter.summarize(total)
        from .gateway.stub import StubModelClient
        used_tiers = {log.tier for log in self.meter.logs}
        m["simulated"] = ((ModelTier.MD in used_tiers and isinstance(self.local.inner, StubModelClient))
                          or (ModelTier.MC in used_tiers and isinstance(self.cloud.inner, StubModelClient)))
        if path == "cache_hit" or not used_tiers:
            m["simulated"] = (isinstance(self.local.inner, StubModelClient)
                              or isinstance(self.cloud.inner, StubModelClient))
        m["ledger_complete"] = error != "call_ledger_write_failed"
        m["privacy"] = self._privacy.ledger.summary()  # 第四指标：披露
        # 请求级落库（tasks 表）：C_time 是请求级量纲，分位数按此表算才不失真
        stages = self.meter.stage_latency
        plan_ms = (stages.get("classify", 0.0) + stages.get("decompose", 0.0)) * 1000
        exec_ms = stages.get("execute", 0.0) * 1000
        try:
            from .gateway import store

            store.record_task(
                run_id=self.run_id, mode=self.mode, question=query[:500],
                answer=answer[:2000], label=label.value, path=path,
                n_subtasks=len(subtasks or []),
                n_local=sum(1 for s in subtasks or [] if s.get("assigned") == "MD"),
                n_cloud=sum(1 for s in subtasks or [] if s.get("assigned") == "MC"),
                total_ms=total * 1000, plan_ms=plan_ms, exec_ms=exec_ms,
                cloud_tokens=m["C_API_total"],
                unknown_usage_calls=m["unknown_usage_calls"], error=error,
            )
        except Exception as e:  # 落库失败不阻断主流程，但要可见
            import sys

            m["ledger_complete"] = False
            print(f"[store] tasks 落库失败: {type(e).__name__}", file=sys.stderr)
            if path.startswith("evidence_"):
                success, error = False, "task_ledger_write_failed"
        return RunResult(
            query=query,
            answer=answer,
            label=label,
            path=path,
            trace=trace,
            metrics=m,
            subtasks=subtasks or [], success=success, error=error,
            provided_sources=provided_sources or [], citations=citations or [],
        )
