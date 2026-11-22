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
from .types import Caller, Label, ModelTier, RunResult, SubTask


class AgentPipeline:
    def __init__(
        self,
        local_client: ModelClient,
        cloud_client: ModelClient,
        meter: CallMeter | None = None,
        theta_default: float = 0.5,
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
        self.scheduler = TaskScheduler(self.local, judge_client=self.cloud)
        self.router = Router(theta_default)
        self.signals = SignalCollector(self.local)
        self.verifier = Verifier()
        self.tools = default_registry()
        self.cache = SemanticCache()

    # ------------------------------------------------------------------ #
    async def run(self, query: str, *, sources: list[dict] | None = None,
                  selector: str = "bm25") -> RunResult:
        """对外请求入口：模型侧异常收敛成失败结果，不让 HTTP 层返回裸 500。

        校验类 ValueError 仍向上抛出（HTTP 层按 400 处理）；模型调用失败
        （截断/空答/连接/HTTP 错误）返回 success=False 且带错误码的结果，
        调用账本里已产生的真实用量照常保留。
        """
        t0 = time.perf_counter()
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
            outcome = await EvidenceWorkflow(self.local, cloud, self.meter).run(query, sources, selector)
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

        # 2. 分类（L1 短路 / L2 兜底）
        meter.stage("classify")
        cls = await self.classifier.classify(query)
        meter.stage_end("classify")
        policy = POLICY[cls.label]
        trace.append(
            f"classify: {cls.label.value} conf={cls.conf:.2f} src={cls.source} "
            f"fast_path={cls.fast_path}（约束={features.constraint_markers} 数学式="
            f"{features.has_math_expr} 选项={features.has_options} 工具词={features.action_verbs}）"
        )

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
        meter.stage("decompose")
        subtasks = await self.decomposer.decompose(query)
        meter.stage_end("decompose")

        if subtasks is None:
            guard.converge_to_cloud()
            trace.append("分解失败 → 整题收敛云端（direct_cloud）")
            answer = (
                await self.cloud.generate(
                    query, max_tokens=512, temperature=0.0, caller=Caller.EXECUTE
                )
            ).text
            return self._finalize(query, answer, cls.label, "direct_cloud", trace, t0)

        trace.append(
            f"分解出 {len(subtasks)} 个子任务: "
            + "; ".join(f"{st.id}:{st.description[:12]}" for st in subtasks)
        )

        meter.stage("execute")
        answers = await self.scheduler.run(
            query, subtasks, self._make_executor(cls, policy, guard, trace, query)
        )
        meter.stage_end("execute")

        # 5. 汇总：DoT 式“最终答案”提示 + 端云分工。
        #    REASONING 只有一个汇点时，它就是要交付的最终答案，直接采用，
        #    避免本地 4B 在二次汇总时抄错或截断；多汇点/其他类型才做汇总。
        meter.stage("summarize")
        key_tasks = [st for st in subtasks if st.key and st.answer.strip()]
        direct_key = (cls.label is Label.REASONING and len(key_tasks) == 1
                      and any(ch.isdigit() for ch in key_tasks[0].answer))
        final_text = ""
        summary_source = ""
        if direct_key:
            final_text = key_tasks[0].answer
            summary_source = f"关键步骤 {key_tasks[0].id}"
            trace.append(f"最终答案直接取唯一关键步骤 {key_tasks[0].id}")
        else:
            summary_prompt = (
                f"原问题: {query}\n\n"
                "已完成子问题及答案（可能出错，仅供核对）:\n"
                + "\n".join(f"{st.id}: {st.description} -> {answers[st.id]}" for st in subtasks)
                + "\n\n请独立解答原问题，不要直接照抄中间结果；"
                "若中间结果与你的计算冲突，以你的计算为准。"
                "\n只输出最终答案（数字或简短结论），不要解释、不要重复推理过程。"
                "\n最终答案:"
            )
            summaries = ([("云端", self.cloud), ("本地", self.local)]
                         if cls.label is Label.REASONING else
                         [("本地", self.local), ("云端", self.cloud)])
            for name, client in summaries:
                try:
                    result = await client.generate(
                        summary_prompt, max_tokens=300, temperature=0.0,
                        caller=Caller.SUMMARIZE,
                    )
                    final_text = result.text
                    summary_source = name
                    trace.append(f"{name}汇总完成")
                    break
                except RuntimeError as error:
                    trace.append(f"{name}汇总失败[{error}] → 尝试下一档")
            if not final_text:  # 两端都不可用：退化为最后一个子任务答案，不丢已得结果
                final_text = answers[subtasks[-1].id] if subtasks else ""
                summary_source = "最后子任务"
                trace.append("两端汇总均失败 → 回退最后子任务答案")
        meter.stage_end("summarize")
        trace.append(f"汇总来源: {summary_source}")
        self.cache.put(query, final_text, features.deterministic)
        subtask_data = [
            {
                "id": st.id,
                "description": st.description,
                "deps": list(st.deps),
                "assigned": st.assigned.value if st.assigned else None,
                "level": st.level,
                "key": st.key,
                "status": st.status,
                "answer": st.answer[:80],
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

    def _make_executor(self, cls, policy, guard: EscalationGuard, trace: list[str],
                       query: str):
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
                        return (
                            await self.cloud.generate(
                                st.description, max_tokens=256,
                                temperature=0.0, caller=Caller.RETRY,
                            )
                        ).text
                    trace.append(f"{st.id} 升级额度耗尽 → 保留失败输出")
                    return tr.output

            # DoT 设计：每个子问题都携带原题与已完成答案，直接前驱单独强调，
            # 避免小模型把子问题当成脱离原题的孤立问句（论文 solve_Sub_Question）。
            # 超预算按完成顺序丢弃最早答案（drop-oldest），原题永不丢弃。
            # 字节预算是上下文装配策略而非计量——计量仍只用后端真实 usage。
            direct = set(st.deps)
            kept = list(ctx)
            total_b = sum(len(ctx[d].encode("utf-8")) for d in kept)
            headroom = (self.context_budget_bytes
                        - len(query.encode("utf-8"))
                        - len(st.description.encode("utf-8")))
            while kept and total_b > headroom:
                dropped = kept.pop(0)
                total_b -= len(ctx[dropped].encode("utf-8"))
                trace.append(f"{st.id} 上下文预算超限，丢弃最早前驱 {dropped}")
            solved = "\n".join(
                f"{d}: {ctx[d]}" + ("（直接相关，重点参考）" if d in direct else "")
                for d in kept
            )
            prompt = (
                f"原问题: {query}\n\n"
                + (f"已完成子问题的答案:\n{solved}\n\n" if solved else "")
                + f"现在只需解决这个子问题: {st.description}\n"
                "请结合原问题给出简洁、可直接使用的答案；不要重复题目，不要解释。"
            )

            # DoT 的关键步骤分配：依赖图汇点直接决定最终答案，优先交给云端。
            # 云端失败时不中止整题，退回本地的信号/升级流程。
            if st.key:
                try:
                    st.assigned = ModelTier.MC
                    trace.append(f"{st.id} 关键步骤（汇点）→ 云端")
                    return (
                        await self.cloud.generate(
                            prompt, max_tokens=512, temperature=0.0,
                            caller=Caller.EXECUTE,
                        )
                    ).text
                except RuntimeError as error:
                    trace.append(f"{st.id} 云端关键步骤失败[{error}] → 本地执行")

            if self.use_signals:
                sig, greedy = await self.signals.collect(prompt)
            else:  # v0 规则协同：无难度信号，路由仅按类型默认倾向
                sig, greedy = None, None
            decision = self.router.assign(label, sig)
            st.assigned = decision.tier
            trace.append(f"{st.id} 路由: {decision.reason}")

            if self.use_signals and greedy is None:
                # 本地采样拿不到有效样本：直接交给云端，不再用同一模型重试同样的失败
                st.assigned = ModelTier.MC
                trace.append(f"{st.id} 本地采样不可用 → 云端执行")
                return (
                    await self.cloud.generate(
                        prompt, max_tokens=512, temperature=0.0, caller=Caller.EXECUTE
                    )
                ).text

            if decision.tier is ModelTier.MD:
                if greedy is None:  # 信号关闭时本地直接生成
                    greedy = await self.local.generate(
                        prompt, max_tokens=256, temperature=0.0, caller=Caller.EXECUTE
                    )
                answer = greedy.text  # 复用 greedy 试答，省一次生成
                ok = self.verifier.verify(verify_kind, st.description, answer)
            else:
                answer = (
                    await self.cloud.generate(
                        prompt, max_tokens=512, temperature=0.0, caller=Caller.EXECUTE
                    )
                ).text
                ok = True

            if not ok and decision.tier is ModelTier.MD:
                if guard.can_escalate():
                    guard.escalate()
                    st.assigned = ModelTier.MC
                    trace.append(f"{st.id} 校验不过 → 升级云端（guard 剩余额度 {guard.remaining}）")
                    answer = (
                        await self.cloud.generate(
                            prompt, max_tokens=512, temperature=0.0, caller=Caller.RETRY
                        )
                    ).text
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
