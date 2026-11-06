/* 端云协同智能体实验台：单次运行、批量测试、记录与统计、记录对比 */
"use strict";

const $ = (id) => document.getElementById(id);
/* 错误码 → 可读提示：后端只回码，人话放在前端，避免改后端契约 */
const ERROR_HINTS = {
  model_incomplete_answer: "本地模型回答被长度上限截断（model_incomplete_answer），已尝试升级云端，可重试或换个问法",
  model_empty_answer: "模型返回空回答（model_empty_answer），可重试",
  model_invalid_response: "模型返回格式异常（model_invalid_response）",
  model_connection_error: "模型接口连接失败（model_connection_error），检查隧道或后端服务",
  model_http_401: "云端鉴权失败（401），检查 API Key",
  model_http_402: "云端余额不足（402）",
  model_http_429: "云端限流（429），请稍后重试",
  model_http_500: "云端服务异常（500）",
  model_http_503: "云端服务不可用（503）",
  evidence_incomplete: "资料问答未完成（引用或格式契约未通过）",
  task_ledger_write_failed: "结果已生成，但指标落库失败（task_ledger_write_failed）",
};
const EXAMPLES = [
  "一个班30人, 每5人一组共多少组? 教室长8米宽6米面积是多少?",
  "下列哪项不是哺乳动物？\nA. 鲸鱼\nB. 鲨鱼\nC. 蝙蝠\nD. 海豚",
  "预算500元以内, 周六出发两日游, 每天不超过3个景点, 请给出方案花费",
  "帮我查询北京明天的天气并发送提醒给同事",
  "帮我计算 12*(3+4) 的结果",
];
const PATH_NAMES = {
  cache_hit: "缓存命中", fast_path: "快速通道·本地直答",
  pipeline: "分解流水线", direct_cloud: "整题收敛云端",
  evidence_bm25: "检索证据 → 云端回答", evidence_local: "本地选证据 → 云端回答",
  evidence_full_fallback: "全文云端回退", evidence_failed: "资料问答未完成",
};

/* ---------------- 标签页 ---------------- */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach((s) => {
    s.hidden = s.id !== "tab-" + name;
  });
  if (name === "records") loadRecords();
}

/* ---------------- 启动 ---------------- */
async function boot() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.onclick = () => switchTab(t.dataset.tab);
  });

  EXAMPLES.forEach((q) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = q.length > 22 ? q.slice(0, 22) + "…" : q;
    chip.onclick = () => { $("sources").value = ""; $("query").value = q; switchTab("run"); run(); };
    $("examples").appendChild(chip);
  });
  const evidenceExample = document.createElement("button");
  evidenceExample.className = "chip";
  evidenceExample.textContent = "资料问答示例";
  evidenceExample.onclick = () => {
    switchTab("run");
    $("query").value = "报名截止时间是什么？";
    $("sources").value = "活动地点在教学楼。\n\n报名截止时间是周五17点。";
    $("selector").value = "bm25";
    document.querySelector(".source-input").open = true;
    run();
  };
  $("examples").appendChild(evidenceExample);

  $("run").onclick = run;
  $("query").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
  });
  $("batch-run").onclick = runBatch;
  $("f-apply").onclick = loadRecords;
  $("f-q").addEventListener("keydown", (e) => { if (e.key === "Enter") loadRecords(); });
  $("diff-run").onclick = runDiff;

  try {
    const runtime = await (await fetch("/api/runtime")).json();
    const stub = runtime.local === "stub" || runtime.cloud === "stub";
    $("runtime").textContent =
      `local: ${runtime.local} · cloud: ${runtime.cloud}` + (stub ? "（stub 演示模式）" : "");
  } catch (e) {
    $("runtime").textContent = "运行时未知";
  }
}

/* ================= 单次运行 ================= */
async function run() {
  const query = $("query").value.trim();
  if (!query) return;
  const payload = { query };
  const sourceText = $("sources").value;
  if (sourceText.trim()) {
    payload.sources = sourceText.split(/\n\s*\n/).filter(text => text.trim())
      .map((text, i) => ({ source_id: "source-" + (i + 1), text }));
    payload.selector = $("selector").value;
  }
  $("run").disabled = true;
  $("answer").textContent = "推理中…";
  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (data.metrics) render(data);
    if (data.error) {
      $("answer").textContent = "运行未完成：" + (ERROR_HINTS[data.error] || data.error);
      $("answer").classList.add("error");
    } else {
      $("answer").classList.remove("error");
    }
  } catch (e) {
    $("answer").textContent = "请求失败，请检查本地服务是否运行。";
  } finally {
    $("run").disabled = false;
  }
}

/* ---------------- 渲染 ---------------- */
function render(d) {
  renderRich($("answer"), d.answer || "");
  $("path").textContent = PATH_NAMES[d.path] || d.path;
  $("record-id").textContent = d.record_id ? `记录 ${d.record_id}` : "";

  const m = d.metrics || {};
  $("t-ctime").textContent = m.C_time_s != null ? `${m.C_time_s.toFixed(3)}s` : "–";
  const stages = m.stage_latency_s || {};
  $("t-ctime-sub").textContent = Object.entries(stages)
    .map(([k, v]) => `${k} ${v.toFixed(3)}`).join(" · ") || "无阶段耗时";
  $("t-capi").textContent = (m.simulated ? "模拟 " : "") + (m.C_API_total ?? "–");
  $("t-capi-sub").textContent = `in ${m.C_API_in ?? 0} / out ${m.C_API_out ?? 0} · 云端调用 ${m.cloud_calls ?? 0} 次`;
  if (m.unknown_usage_calls) $("t-capi-sub").textContent += " · 已知小计，另有 " + m.unknown_usage_calls + " 次用量未知";
  if (m.simulated) $("t-capi-sub").textContent += " · 模拟用量，不代表真实模型表现";
  if (m.ledger_complete === false) $("t-capi-sub").textContent += " · 账本不完整";
  $("source-trace").textContent = (d.provided_sources || []).length
    ? "提供给云端的来源：" + d.provided_sources.join("、") + "；答案引用：" + ((d.citations || []).join("、") || "无")
    : "";
  $("t-local").textContent = m.local_calls ?? "–";
  $("t-local-sub").textContent = `本地推理 ${m.local_latency_s ?? 0}s`;
  $("t-label").textContent = d.label;
  $("t-label-sub").textContent = `路径: ${d.path}`;

  renderPrivacy(m.privacy, d.subtasks || []);
  renderDag(d.subtasks || []);
  renderTrace(d.trace || []);
  renderStages(m);
  renderCalls(m);
  renderSubtasks(d);
}

/* ---------------- 隐私执行明细 ---------------- */
function renderPrivacy(p, subtasks) {
  const card = $("privacy-card");
  if (!card) return;
  if (!p || !p.enabled) {
    // 隐私关闭（消融对照）也要可见：这本身是实验状态，不能悄悄藏掉
    card.style.display = "";
    $("priv-tiles").innerHTML =
      '<div class="muted">隐私模块已关闭（DOT_PRIVACY=off），全部云端转交为明文，' +
      '该组数据用作消融对照。</div>';
    $("priv-views").innerHTML = '<div class="muted">未启用</div>';
    $("t-priv").textContent = "关";
    $("t-priv-sub").textContent = "隐私模块未启用";
    return;
  }
  const mc = p.mode_counts || {};
  const tiles = [
    ["判别模式（HIGH）", mc.cloud_judge ?? 0, "云端只见假名化候选"],
    ["假名化转交（LOW）", mc.cloud_masked ?? 0, "转交前替换，答案回填"],
    ["明文转交（NONE）", mc.cloud_plain ?? 0, "无敏感内容，旧行为"],
    ["本地执行", mc.local_only ?? 0, "云端不参与"],
    ["替换实体数", p.entities_masked ?? 0, `泄露 ${p.leak_count ?? 0} 次（期望 0）`],
    ["回填校验失败", p.roundtrip_failures ?? 0, "假名残留或实体数溢出"],
  ];
  $("priv-tiles").innerHTML = tiles.map(([label, value, sub]) => {
    const warn = (label === "替换实体数" && (p.leak_count ?? 0) > 0)
      || (label === "回填校验失败" && value > 0);
    return `<div class="card tile"><div class="label">${escapeHtml(label)}</div>` +
      `<div class="value" style="font-size:22px;color:${warn ? "var(--mc)" : "inherit"}">${value}</div>` +
      `<div class="sub">${escapeHtml(sub)}</div></div>`;
  }).join("");
  const views = (p.views || []).length ? p.views
    : subtasks.filter((st) => st.cloud_view).map((st) => st.cloud_view);
  $("priv-views").innerHTML = views.length ? views.map((v) =>
    `<div class="sub-view"><span class="tag cloud">${escapeHtml(v.id || "–")} · ${escapeHtml(v.mode)}</span>` +
    `<div class="sub-a">${escapeHtml(v.text || "")}</div></div>`).join("")
    : '<div class="muted">本次无云端转交</div>';

  // 指标 tile：披露计数 = 明文实体 + 泄露
  const disclosed = (p.entities_disclosed_plain ?? 0) + (p.leak_count ?? 0);
  $("t-priv").textContent = disclosed;
  $("t-priv-sub").textContent = `明文披露 ${p.entities_disclosed_plain ?? 0} · 泄露 ${p.leak_count ?? 0} · 替换 ${p.entities_masked ?? 0}`;
}

/* ---------------- 依赖图（拓扑分层 SVG）---------------- */
function renderDag(subtasks) {
  const box = $("dag");
  if (!subtasks.length) {
    box.innerHTML = '<div class="dag-empty">本请求未拆分任务。资料问答的证据选择与引用可在结果和决策时间线中查看。</div>';
    return;
  }
  const levels = {};
  subtasks.forEach((st) => { (levels[st.level] = levels[st.level] || []).push(st); });
  const levelKeys = Object.keys(levels).map(Number).sort((a, b) => a - b);

  const W = 168, H = 74, GX = 56, GY = 26, PAD = 8;
  const maxPerLevel = Math.max(...levelKeys.map((k) => levels[k].length));
  const height = maxPerLevel * (H + GY) - GY + PAD * 2;
  const width = levelKeys.length * (W + GX) - GX + PAD * 2;
  const pos = {};
  levelKeys.forEach((lv, ci) => {
    const arr = levels[lv];
    arr.forEach((st, ri) => {
      const gap = (height - arr.length * H) / (arr.length + 1);
      pos[st.id] = { x: PAD + ci * (W + GX), y: gap * (ri + 1) + ri * H, st };
    });
  });

  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  let svg = `<svg width="${width}" height="${height}" role="img" aria-label="子任务依赖图">`;
  subtasks.forEach((st) => {
    (st.deps || []).forEach((dep) => {
      if (!pos[dep] || !pos[st.id]) return;
      const a = pos[dep], b = pos[st.id];
      const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2;
      const mx = (x1 + x2) / 2;
      svg += `<path class="edge" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}"/>`;
    });
  });
  subtasks.forEach((st) => {
    const p = pos[st.id];
    const cls = st.assigned === "MC" ? "mc" : "md";
    const tip = esc(`【${st.id} → ${st.assigned}】${st.description}\n答案: ${st.answer || "—"}`);
    const desc = st.description.length > 16 ? st.description.slice(0, 16) + "…" : st.description;
    svg += `<g class="node ${cls}"><title>${tip}</title>
      <rect class="${cls}" x="${p.x}" y="${p.y}" width="${W}" height="${H}" rx="8"/>
      <text class="sid" x="${p.x + 12}" y="${p.y + 20}">${st.id} · ${st.assigned}
        <tspan class="lv" fill="#898781">· L${st.level}</tspan></text>
      <text class="desc" x="${p.x + 12}" y="${p.y + 42}">${esc(desc)}</text>
      <text class="desc" x="${p.x + 12}" y="${p.y + 60}" fill="#898781">${esc((st.answer || "—").slice(0, 18))}</text>
    </g>`;
  });
  svg += "</svg>";
  box.innerHTML = svg;
}

/* ---------------- 富文本：Markdown + LaTeX ---------------- */
function escapeHtml(str) {
  return String(str == null ? "" : str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function sanitizeHtml(html) {
  return html
    .replace(/<\s*(script|style)[^>]*>[\s\S]*?<\s*\/\s*\1\s*>/gi, "")
    .replace(/\son[a-z]+\s*=\s*"[^"]*"/gi, "")
    .replace(/\son[a-z]+\s*=\s*'[^']*'/gi, "")
    .replace(/javascript:/gi, "");
}

function normalizeMath(src) {
  return String(src).split("\n").map((line) => {
    const m = line.match(/^(.*?)(\\+)[ \t]*$/);
    if (!m) return line;
    return m[2].length % 2 === 1 ? m[1] + m[2] + "\\" : line;
  }).join("\n");
}

function katexHtml(item) {
  if (!window.katex || typeof window.katex.renderToString !== "function") {
    return `<code>${escapeHtml(item.src)}</code>`;
  }
  try {
    return window.katex.renderToString(item.src, {
      displayMode: item.display,
      throwOnError: false,
      strict: false,
    });
  } catch (e) {
    return `<code>${escapeHtml(item.src)}</code>`;
  }
}

function renderRich(el, text) {
  const src = String(text == null ? "" : text);
  if (!src.trim()) { el.textContent = "—"; return; }
  el.classList.add("md");

  const items = [];
  const keep = (body, display) => {
    items.push({ src: normalizeMath(body.trim()), display });
    return `@@MATH${items.length - 1}@@`;
  };

  const protectedText = src
    .replace(/\$\$([\s\S]+?)\$\$/g, (m, g) => keep(g, true))
    .replace(/\\\[([\s\S]+?)\\\]/g, (m, g) => keep(g, true))
    .replace(/\\\(([\s\S]+?)\\\)/g, (m, g) => keep(g, false))
    .replace(/\$([^$\n]+?)\$/g, (m, g) => keep(g, false));

  let html = protectedText;
  if (window.marked && typeof window.marked.parse === "function") {
    html = sanitizeHtml(window.marked.parse(protectedText, { breaks: true }));
  } else {
    html = escapeHtml(protectedText).replace(/\n/g, "<br>");
  }

  html = html.replace(/<p>\s*@@MATH(\d+)@@\s*<\/p>/g, (m, i) => {
    const item = items[Number(i)];
    return item && item.display ? `<div class="math-block">${katexHtml(item)}</div>` : m;
  });
  html = html.replace(/@@MATH(\d+)@@/g, (m, i) => {
    const item = items[Number(i)];
    return item ? katexHtml(item) : m;
  });

  el.innerHTML = html;
}

/* ---------------- 阶段耗时 ---------------- */
function renderStages(m) {
  const el = $("stages");
  if (!el) return;
  const rows = Object.entries(m.stage_latency_s || {})
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);
  if (!rows.length) { el.innerHTML = '<div class="muted">本次请求没有阶段计时</div>'; return; }
  const max = Math.max(...rows.map(([, v]) => v));
  el.innerHTML = rows.map(([k, v]) => {
    const pct = Math.max(2, Math.round((v / max) * 100));
    return `<div class="stage-row"><span class="stage-name">${escapeHtml(k)}</span>` +
      `<span class="bar"><i style="width:${pct}%"></i></span>` +
      `<span class="stage-val">${v.toFixed(3)}s</span></div>`;
  }).join("");
}

/* ---------------- 逐次调用 ---------------- */
function renderCalls(m) {
  const el = $("calls");
  if (!el) return;
  const calls = m.calls || [];
  const cnt = $("callcount");
  if (cnt) cnt.textContent = calls.length ? `（共 ${calls.length} 次）` : "";
  if (!calls.length) {
    el.innerHTML = '<tbody><tr><td class="muted">本次请求没有模型调用（可能命中缓存）</td></tr></tbody>';
    return;
  }
  const head = "<thead><tr><th>#</th><th>调用方</th><th>档位</th><th>输入</th><th>输出</th><th>耗时</th></tr></thead>";
  const body = calls.map((c) => {
    const tierCls = c.tier === "MC" ? "cloud" : "local";
    const down = c.requested_tier && c.requested_tier !== c.tier
      ? ' <span class="tag warn">降级</span>' : "";
    const tok = (v) => (v == null ? "未知" : v);
    return `<tr><td>${c.seq}</td><td>${escapeHtml(c.caller)}</td>` +
      `<td><span class="tag ${tierCls}">${c.tier}</span>${down}</td>` +
      `<td class="num">${tok(c.in)}</td><td class="num">${tok(c.out)}</td>` +
      `<td class="num">${(c.latency_s || 0).toFixed(2)}s</td></tr>`;
  }).join("");
  el.innerHTML = head + `<tbody>${body}</tbody>`;
}

/* ---------------- 子任务明细 ---------------- */
function renderSubtasks(d) {
  const el = $("subdetail");
  if (!el) return;
  const subs = d.subtasks || [];
  if (!subs.length) {
    el.innerHTML = '<div class="muted">本请求未拆分任务（快速通道或整题收敛）</div>';
    return;
  }
  el.innerHTML = subs.map((st) => {
    const deps = (st.deps || []).length ? st.deps.join("、") : "无";
    const tierCls = st.assigned === "MC" ? "cloud" : "local";
    const view = st.cloud_view
      ? `<div class="sub-view"><span class="tag cloud">云端所见（${escapeHtml(st.cloud_view.mode)}）</span>
         <div class="sub-a">${escapeHtml(st.cloud_view.text || "")}</div></div>`
      : "";
    return `<div class="sub-item">
      <div class="sub-head"><b>${escapeHtml(st.id)}</b>
        <span class="tag ${tierCls}">${escapeHtml(st.assigned || "–")}</span>
        <span class="muted">L${st.level} · 依赖 ${escapeHtml(deps)}${st.status ? " · " + escapeHtml(st.status) : ""}</span></div>
      <div class="sub-q">${escapeHtml(st.description || "")}</div>
      <div class="sub-a">${escapeHtml(st.answer || "")}</div>
      ${view}
    </div>`;
  }).join("");
}

/* ---------------- 时间线 ---------------- */
function renderTrace(trace) {
  $("trace").innerHTML = (trace || []).map((line) => {
    const text = typeof line === "string" ? line : JSON.stringify(line);
    let cls = "";
    if (/云端|升级|MC/.test(text)) cls = "cloud";
    else if (/本地|MD|工具执行成功/.test(text)) cls = "local";
    else if (/失败|耗尽|重分类|收敛/.test(text)) cls = "warn";
    return `<li class="${cls}">${escapeHtml(text)}</li>`;
  }).join("");
}

/* ================= 批量测试 ================= */
async function runBatch() {
  const queries = $("batch-queries").value.split("\n")
    .map((q) => q.trim()).filter(Boolean);
  if (!queries.length) return;
  // 逐条提交而不是整批发给 /api/batch：每条完成即更新进度，失败不拖垮整批
  $("batch-run").disabled = true;
  const results = [];
  const start = Date.now();
  for (let i = 0; i < queries.length; i++) {
    const elapsed = ((Date.now() - start) / 1000).toFixed(0);
    $("batch-progress").textContent =
      `执行中 ${i + 1}/${queries.length} · 已用时 ${elapsed}s`;
    try {
      const resp = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queries[i] }),
      });
      const data = await resp.json();
      results.push({
        query: queries[i], record_id: data.record_id,
        label: data.label, path: data.path,
        success: data.success, error: data.error,
        c_time_s: data.metrics ? data.metrics.C_time_s : null,
        cloud_tokens: data.metrics ? data.metrics.C_API_total : null,
        cloud_calls: data.metrics ? data.metrics.cloud_calls : 0,
        local_calls: data.metrics ? data.metrics.local_calls : 0,
      });
    } catch (e) {
      results.push({ query: queries[i], success: false, error: "request_failed" });
    }
    renderBatchTable(results);  // 每条完成即刷新表格，边跑边看
  }
  const okCount = results.filter((r) => r.success).length;
  const totalSec = ((Date.now() - start) / 1000).toFixed(1);
  $("batch-progress").textContent =
    `完成 ${queries.length} 条，成功 ${okCount} 条 · 总用时 ${totalSec}s · 已逐条落盘，见"记录与统计"`;
  $("batch-run").disabled = false;
}

function renderBatchTable(results) {
  const el = $("batch-table");
  if (!results || !results.length) {
    el.innerHTML = '<tbody><tr><td class="muted">无结果</td></tr></tbody>';
    return;
  }
  const head = "<thead><tr><th>#</th><th>查询</th><th>类型</th><th>路径</th><th>状态</th>" +
    "<th>耗时</th><th>云端 token</th><th>本地/云端调用</th><th></th></tr></thead>";
  const body = results.map((r, i) => {
    const status = r.success
      ? '<span class="tag local">成功</span>'
      : `<span class="tag warn">${escapeHtml(r.error || "失败")}</span>`;
    return `<tr><td>${i + 1}</td><td>${escapeHtml((r.query || "").slice(0, 40))}</td>` +
      `<td>${escapeHtml(r.label || "–")}</td><td>${escapeHtml(PATH_NAMES[r.path] || r.path || "–")}</td>` +
      `<td>${status}</td>` +
      `<td class="num">${r.c_time_s != null ? r.c_time_s.toFixed(2) + "s" : "–"}</td>` +
      `<td class="num">${r.cloud_tokens ?? "–"}</td>` +
      `<td class="num">${r.local_calls ?? "–"} / ${r.cloud_calls ?? "–"}</td>` +
      `<td>${r.record_id ? `<a href="#" onclick="openRecord('${r.record_id}');return false">查看</a>` : ""}</td></tr>`;
  }).join("");
  el.innerHTML = head + `<tbody>${body}</tbody>`;
}

/* ================= 记录与统计 ================= */
let RECORD_ROWS = [];

async function loadRecords() {
  const params = new URLSearchParams();
  if ($("f-label").value) params.set("label", $("f-label").value);
  if ($("f-path").value) params.set("path", $("f-path").value);
  if ($("f-success").value) params.set("success", $("f-success").value);
  if ($("f-q").value.trim()) params.set("q", $("f-q").value.trim());
  params.set("limit", "200");
  const [runsData, statsData] = await Promise.all([
    fetch("/api/runs?" + params.toString()).then((r) => r.json()),
    fetch("/api/stats" + ($("f-label").value ? "?label=" + $("f-label").value : "")).then((r) => r.json()),
  ]);
  RECORD_ROWS = runsData.runs || [];
  renderStats(statsData);
  renderRunsTable();
  renderDiffSelectors();
}

function fillSelect(select, counts, allLabel) {
  if (!select || !counts) return;
  const current = select.value;
  const options = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<option value="${escapeHtml(k)}">${escapeHtml(k)}（${v}）</option>`).join("");
  select.innerHTML = `<option value="">${escapeHtml(allLabel)}</option>` + options;
  select.value = current;  // 保留用户已选的筛选
  if (select.value !== current) select.value = "";  // 选项消失时回落到全部
}

function renderStats(s) {
  const el = $("stats-tiles");
  // 筛选下拉从实际数据填充（类型/路径选项随记录变化）
  fillSelect($("f-label"), s.label_counts, "全部类型");
  fillSelect($("f-path"), s.path_counts, "全部路径");
  if (!s || !s.total) {
    el.innerHTML = '<div class="muted">还没有落盘记录，先去"运行"或"批量测试"跑几条。</div>';
    $("stats-dist").textContent = "";
    return;
  }
  const tiles = [
    ["记录总数", s.total, `成功 ${s.success}（${s.success_rate != null ? (s.success_rate * 100).toFixed(1) + "%" : "–"}）`],
    ["平均响应时间", s.c_time_avg != null ? s.c_time_avg.toFixed(3) + "s" : "–",
      `P50 ${s.c_time_p50 != null ? s.c_time_p50.toFixed(3) : "–"}s · P95 ${s.c_time_p95 != null ? s.c_time_p95.toFixed(3) : "–"}s`],
    ["云端 token 总计", s.cloud_tokens_total ?? 0, `平均 ${s.cloud_tokens_avg ?? "–"} / 条`],
    ["模拟运行", s.simulated_runs, "stub 模式下跑出的记录，数字不代表真实模型"],
  ];
  el.innerHTML = tiles.map(([label, value, sub]) =>
    `<div class="card tile"><div class="label">${escapeHtml(label)}</div>` +
    `<div class="value" style="font-size:22px">${escapeHtml(String(value))}</div>` +
    `<div class="sub">${escapeHtml(sub)}</div></div>`).join("");
  const dist = Object.entries(s.label_counts || {}).map(([k, v]) => `${k}×${v}`).join(" · ")
    + "　｜　" + Object.entries(s.path_counts || {}).map(([k, v]) => `${k}×${v}`).join(" · ");
  $("stats-dist").textContent = dist;
}

function renderRunsTable() {
  const el = $("runs-table");
  if (!RECORD_ROWS.length) {
    el.innerHTML = '<tbody><tr><td class="muted">当前筛选条件下没有记录</td></tr></tbody>';
    return;
  }
  const head = "<thead><tr><th>时间</th><th>查询</th><th>类型</th><th>路径</th><th>状态</th>" +
    "<th>耗时</th><th>云端 token</th><th></th></tr></thead>";
  const body = RECORD_ROWS.map((r) => {
    const status = r.success
      ? '<span class="tag local">成功</span>'
      : `<span class="tag warn">${escapeHtml(r.error || "失败")}</span>`;
    return `<tr><td class="muted">${escapeHtml(r.ts)}</td>` +
      `<td>${escapeHtml((r.query || "").slice(0, 36))}</td>` +
      `<td>${escapeHtml(r.label)}</td><td>${escapeHtml(r.path)}</td><td>${status}</td>` +
      `<td class="num">${r.c_time_s != null ? r.c_time_s.toFixed(2) + "s" : "–"}</td>` +
      `<td class="num">${r.cloud_tokens ?? "–"}</td>` +
      `<td><a href="#" onclick="openRecord('${r.id}');return false">查看</a></td></tr>`;
  }).join("");
  el.innerHTML = head + `<tbody>${body}</tbody>`;
}

async function openRecord(id) {
  const resp = await fetch("/api/runs/" + id);
  if (!resp.ok) return;
  const record = await resp.json();
  switchTab("run");
  $("query").value = record.query || "";
  render(record);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderDiffSelectors() {
  const options = RECORD_ROWS.map((r) =>
    `<option value="${r.id}">${escapeHtml(r.ts)} · ${escapeHtml((r.query || "").slice(0, 16))}</option>`).join("");
  const keepA = $("diff-a").value, keepB = $("diff-b").value;
  $("diff-a").innerHTML = "<option value=''>记录 A…</option>" + options;
  $("diff-b").innerHTML = "<option value=''>记录 B…</option>" + options;
  $("diff-a").value = keepA;
  $("diff-b").value = keepB;
}

async function runDiff() {
  const a = $("diff-a").value, b = $("diff-b").value;
  const out = $("diff-out");
  if (!a || !b || a === b) {
    out.innerHTML = '<div class="muted">请选择两条不同的记录。</div>';
    return;
  }
  const [ra, rb] = await Promise.all([
    fetch("/api/runs/" + a).then((r) => r.json()),
    fetch("/api/runs/" + b).then((r) => r.json()),
  ]);
  const mixed = Boolean(ra.simulated) !== Boolean(rb.simulated);
  out.innerHTML =
    (mixed ? '<div class="muted" style="color:var(--mc)">注意：A 和 B 一条是 stub 模拟记录、一条是真实记录，数字不可直接对比。</div>' : "")
    + renderDiff(ra, rb);
}

function metricRow(name, va, vb, fmt) {
  const f = fmt || ((x) => x);
  const a = va == null ? "–" : f(va);
  const b = vb == null ? "–" : f(vb);
  const numA = typeof va === "number" ? va : null;
  const numB = typeof vb === "number" ? vb : null;
  let mark = "";
  if (numA != null && numB != null && numA !== numB) {
    const delta = ((numB - numA) / numA * 100);
    mark = `<span class="muted">${delta > 0 ? "+" : ""}${delta.toFixed(1)}%</span>`;
  }
  return `<tr><td>${escapeHtml(name)}</td><td class="num">${escapeHtml(String(a))}</td>` +
    `<td class="num">${escapeHtml(String(b))}</td><td>${mark}</td></tr>`;
}

function renderDiff(ra, rb) {
  const ma = ra.metrics || {}, mb = rb.metrics || {};
  const rows = [
    metricRow("完整响应时间", ra.c_time_s ?? ma.C_time_s, rb.c_time_s ?? mb.C_time_s,
      (x) => Number(x).toFixed(3) + "s"),
    metricRow("云端 token", ra.cloud_tokens ?? ma.C_API_total, rb.cloud_tokens ?? mb.C_API_total),
    metricRow("云端调用次数", ma.cloud_calls, mb.cloud_calls),
    metricRow("本地调用次数", ma.local_calls, mb.local_calls),
    metricRow("替换实体数", (ma.privacy || {}).entities_masked, (mb.privacy || {}).entities_masked),
    metricRow("明文披露实体", (ma.privacy || {}).entities_disclosed_plain, (mb.privacy || {}).entities_disclosed_plain),
    metricRow("泄露（期望 0）", (ma.privacy || {}).leak_count, (mb.privacy || {}).leak_count),
  ];
  // 子任务分配对比（按 id 对齐）
  const sa = ra.subtasks || [], sb = rb.subtasks || [];
  const ids = [...new Set([...sa.map((s) => s.id), ...sb.map((s) => s.id)])];
  const assignRows = ids.map((id) => {
    const ta = (sa.find((s) => s.id === id) || {}).assigned || "–";
    const tb = (sb.find((s) => s.id === id) || {}).assigned || "–";
    const changed = ta !== tb;
    return `<tr><td>${escapeHtml(id)}</td>` +
      `<td class="num">${escapeHtml(ta)}</td><td class="num">${escapeHtml(tb)}</td>` +
      `<td>${changed ? '<span class="tag warn">分配不同</span>' : ""}</td></tr>`;
  }).join("");
  return `<table class="calls">
    <thead><tr><th></th><th>A ${escapeHtml(ra.id || "")}</th><th>B ${escapeHtml(rb.id || "")}</th><th></th></tr></thead>
    <tbody>${rows.join("")}</tbody></table>
    <div class="muted" style="margin:8px 0 4px">A: ${escapeHtml(ra.query || "")}</div>
    <div class="muted" style="margin-bottom:8px">B: ${escapeHtml(rb.query || "")}</div>
    <h3>子任务分配对比</h3>
    <table class="calls"><thead><tr><th>子任务</th><th>A</th><th>B</th><th></th></tr></thead>
    <tbody>${assignRows}</tbody></table>`;
}

boot();
