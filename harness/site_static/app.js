let MANIFEST = null;
let DATA = null;
let currentView = "leaderboard";
let lastListView = "leaderboard";
let routeSeq = 0;

const app = document.querySelector("#app");
const VALID_VIEWS = new Set(["leaderboard", "problems", "models", "detail"]);
const runCache = new Map();
const problemCache = new Map();
const detailCache = new Map();

function defaultRoute() {
  return {
    runId: MANIFEST?.default_run_id || MANIFEST?.runs?.[0]?.run_id || "",
    view: "leaderboard",
    problemId: null,
    modelName: null,
    sampleId: null,
  };
}

function parseHash() {
  const route = defaultRoute();
  const encoded = (location.hash || "").replace(/^#\/?/, "");
  const parts = encoded.split("/").filter(Boolean).map((part) => {
    try { return decodeURIComponent(part); } catch { return part; }
  });
  if (parts[0]) route.runId = parts[0];
  if (parts[1] === "detail") {
    route.view = "detail";
    route.modelName = parts[2] || null;
    route.sampleId = parts[3] || null;
    return route;
  }
  if (parts[1] && VALID_VIEWS.has(parts[1])) route.view = parts[1];
  if (route.view === "problems") route.problemId = parts[2] || null;
  if (route.view === "models") route.modelName = parts[2] || null;
  return route;
}

function routeHash(route) {
  const runId = encodeURIComponent(route.runId || defaultRoute().runId);
  if (route.view === "detail" && route.modelName && route.sampleId) {
    return `#/${runId}/detail/${encodeURIComponent(route.modelName)}/${encodeURIComponent(route.sampleId)}`;
  }
  if (route.view === "problems") {
    return route.problemId
      ? `#/${runId}/problems/${encodeURIComponent(route.problemId)}`
      : `#/${runId}/problems`;
  }
  if (route.view === "models") {
    return route.modelName
      ? `#/${runId}/models/${encodeURIComponent(route.modelName)}`
      : `#/${runId}/models`;
  }
  return `#/${runId}/leaderboard`;
}

function navigate(patch, { replace = false } = {}) {
  const next = { ...parseHash(), ...patch };
  const hash = routeHash(next);
  if (hash === location.hash) {
    applyRoute(next);
    return;
  }
  if (replace) {
    history.replaceState(null, "", hash);
    applyRoute(next);
    return;
  }
  location.hash = hash;
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function inlineMarkdown(text) {
  return esc(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function renderMarkdown(source) {
  const lines = String(source || "").replaceAll("\r\n", "\n").split("\n");
  const html = [];
  let listStack = [];

  function closeLists(toLevel = 0) {
    while (listStack.length > toLevel) {
      html.push("</ul>");
      listStack.pop();
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/, "");
    if (!line.trim()) {
      closeLists(0);
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      closeLists(0);
      const level = heading[1].length;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const listMatch = line.match(/^(\s*)-\s+(.*)$/);
    if (listMatch) {
      const indent = listMatch[1].replaceAll("\t", "  ").length;
      const level = Math.floor(indent / 2) + 1;
      while (listStack.length < level) {
        html.push("<ul>");
        listStack.push(true);
      }
      closeLists(level);
      html.push(`<li>${inlineMarkdown(listMatch[2])}</li>`);
      continue;
    }

    closeLists(0);
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  closeLists(0);
  return html.join("");
}

function openLightbox(src, alt) {
  const lightbox = document.querySelector("#lightbox");
  const image = document.querySelector("#lightbox-image");
  if (!lightbox || !image) return;
  image.src = src;
  image.alt = alt || "";
  lightbox.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  const lightbox = document.querySelector("#lightbox");
  const image = document.querySelector("#lightbox-image");
  if (!lightbox || !image) return;
  lightbox.hidden = true;
  image.removeAttribute("src");
  document.body.style.overflow = "";
}

function bindLightbox(root = document) {
  root.querySelectorAll("img.problem-image[data-zoom]").forEach((image) => {
    image.addEventListener("click", () => openLightbox(image.src, image.alt));
  });
}

function pct(value) {
  return `${Math.round((value || 0) * 1000) / 10}%`;
}

function money(value) {
  return value ? `$${Number(value).toFixed(4)}` : "-";
}

function statusPill(passed, judged) {
  if (!judged || passed === null || passed === undefined) return `<span class="pill warn">Unjudged</span>`;
  return passed ? `<span class="pill pass">Pass</span>` : `<span class="pill fail">Fail</span>`;
}

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function runLabel(runId) {
  const run = (MANIFEST?.runs || []).find((item) => item.run_id === runId);
  return run?.label || runId || "";
}

function isGrepRules(runId) {
  return runId === "grep-rules" || runId === "tools-rules";
}

function showToolCalls(row) {
  return isGrepRules(row.version || DATA.run_id || "");
}

function rolloutUsage(row) {
  const parts = [`${formatCount(row.output_tokens)} output tokens`];
  if (showToolCalls(row)) parts.push(`${formatCount(row.tool_call_count)} tool calls`);
  return parts.join(" · ");
}

function modelVersionName(modelName, runId) {
  return `${modelName} @ ${runLabel(runId)}`;
}

function sampleSortKey(sampleId) {
  const asInt = Number.parseInt(sampleId, 10);
  return Number.isNaN(asInt) ? [10 ** 9, sampleId] : [asInt, sampleId];
}

function emptyAttempt(model, sampleId) {
  return {
    model: model.name,
    original_model: model.original_model,
    version: model.version,
    sample_id: sampleId,
    status: "not_attempted",
    passed: null,
    judged: false,
    output_tokens: 0,
    tool_call_count: 0,
    rollout_cost_usd: 0,
    judge_cost_usd: 0,
  };
}

function alignProblemModels(problems, leaderboard) {
  for (const problem of problems) {
    const byModel = Object.fromEntries((problem.models || []).map((row) => [row.model, row]));
    const aligned = leaderboard.map((model) => byModel[model.name] || emptyAttempt(model, problem.id));
    problem.models = aligned;
    const judged = aligned.filter((row) => row.passed !== null && row.passed !== undefined);
    problem.judged = judged.length;
    problem.passed = judged.filter((row) => row.passed).length;
    problem.attempted = aligned.filter((row) => row.status !== "not_attempted").length;
  }
}

function combineRunSummaries(runs) {
  const leaderboard = [];
  const models = [];
  const problemsById = {};

  for (const run of runs) {
    const runId = run.run_id;
    for (const row of run.leaderboard || []) {
      leaderboard.push({
        ...row,
        name: modelVersionName(row.name, runId),
        original_model: row.name,
        version: runId,
      });
    }
    for (const model of run.models || []) {
      const displayName = modelVersionName(model.name, runId);
      models.push({
        ...model,
        name: displayName,
        original_model: model.name,
        version: runId,
        problems: (model.problems || []).map((summary) => ({
          ...summary,
          model: displayName,
          original_model: model.name,
          version: runId,
        })),
      });
    }
    for (const problem of run.problems || []) {
      const combined = problemsById[problem.id] || {
        id: problem.id,
        difficulty: problem.difficulty || "Unknown",
        has_detail: Boolean(problem.has_detail),
        image: problem.image || null,
        models: [],
      };
      if (!combined.image && problem.image) combined.image = problem.image;
      if (problem.has_detail) combined.has_detail = true;
      combined.models.push(
        ...(problem.models || []).map((summary) => ({
          ...summary,
          model: modelVersionName(summary.model, runId),
          original_model: summary.model,
          version: runId,
        })),
      );
      problemsById[problem.id] = combined;
    }
  }

  leaderboard.sort((a, b) => (b.pass_rate - a.pass_rate) || (b.passed - a.passed) || a.version.localeCompare(b.version) || a.name.localeCompare(b.name));
  leaderboard.forEach((row, index) => { row.rank = index + 1; });
  const modelsByName = Object.fromEntries(models.map((model) => [model.name, model]));
  const orderedModels = leaderboard.map((row) => modelsByName[row.name]).filter(Boolean);
  const problems = Object.values(problemsById).sort((a, b) => {
    const [ai, as] = sampleSortKey(a.id);
    const [bi, bs] = sampleSortKey(b.id);
    return ai - bi || as.localeCompare(bs);
  });
  alignProblemModels(problems, leaderboard);
  return {
    run_id: "all-versions",
    label: "All versions",
    leaderboard,
    problems,
    models: orderedModels,
  };
}

function manifestRun(runId) {
  return (MANIFEST?.runs || []).find((item) => item.run_id === runId) || MANIFEST?.runs?.[0] || null;
}

function realRuns() {
  return (MANIFEST?.runs || []).filter((item) => !item.virtual);
}

function assetUrl(path) {
  const stamp = MANIFEST?.generated_at;
  if (!stamp) return path;
  return `${path}${path.includes("?") ? "&" : "?"}v=${encodeURIComponent(stamp)}`;
}

async function fetchJson(path) {
  const response = await fetch(assetUrl(path));
  if (!response.ok) {
    throw new Error(`${path} (${response.status})`);
  }
  return response.json();
}

async function loadRunSummary(runId) {
  if (runCache.has(runId)) return runCache.get(runId);
  const pending = fetchJson(`data/runs/${encodeURIComponent(runId)}/summary.json`);
  runCache.set(runId, pending);
  try {
    const data = await pending;
    runCache.set(runId, data);
    return data;
  } catch (error) {
    runCache.delete(runId);
    throw error;
  }
}

async function loadRun(runId) {
  const meta = manifestRun(runId);
  if (meta?.virtual || runId === "all-versions") {
    if (runCache.has("all-versions")) return runCache.get("all-versions");
    const summaries = await Promise.all(realRuns().map((run) => loadRunSummary(run.run_id)));
    const combined = combineRunSummaries(summaries);
    runCache.set("all-versions", combined);
    return combined;
  }
  return loadRunSummary(runId);
}

async function loadProblem(problemId) {
  if (problemCache.has(problemId)) return problemCache.get(problemId);
  try {
    const data = await fetchJson(`data/problems/${encodeURIComponent(problemId)}.json`);
    problemCache.set(problemId, data);
    return data;
  } catch {
    const missing = { id: problemId, missing: true };
    problemCache.set(problemId, missing);
    return missing;
  }
}

function findAttempt(modelName, sampleId) {
  for (const problem of DATA?.problems || []) {
    const row = (problem.models || []).find((item) => item.model === modelName && item.sample_id === sampleId);
    if (row) return row;
  }
  for (const model of DATA?.models || []) {
    if (model.name !== modelName) continue;
    const row = (model.problems || []).find((item) => item.sample_id === sampleId);
    if (row) return row;
  }
  return null;
}

function detailPathFor(modelName, sampleId) {
  const row = findAttempt(modelName, sampleId);
  if (row?.detail_path) return row.detail_path;
  const version = row?.version || (DATA?.run_id !== "all-versions" ? DATA.run_id : null);
  const model = row?.original_model || modelName;
  if (!version) return null;
  return `data/runs/${version}/${model}/${sampleId}.json`;
}

async function loadDetail(modelName, sampleId) {
  const path = detailPathFor(modelName, sampleId);
  if (!path) return null;
  if (detailCache.has(path)) return detailCache.get(path);
  try {
    const data = await fetchJson(path);
    detailCache.set(path, data);
    return data;
  } catch {
    detailCache.set(path, null);
    return null;
  }
}

function initRunSelector() {
  const selector = document.querySelector("#run-select");
  selector.innerHTML = (MANIFEST.runs || [])
    .map((run) => `<option value="${esc(run.run_id)}">${esc(run.label || run.run_id)}</option>`)
    .join("");
  selector.disabled = (MANIFEST.runs || []).length <= 1;
  selector.addEventListener("change", () => navigate({ runId: selector.value }));
}

async function applyRoute(route = parseHash()) {
  const seq = ++routeSeq;
  const meta = manifestRun(route.runId);
  try {
    DATA = await loadRun(meta?.run_id || route.runId);
  } catch (error) {
    if (seq !== routeSeq) return;
    DATA = null;
    app.innerHTML = `<section class="card"><h2>Could not load run data</h2><pre>${esc(error.stack || error.message || error)}</pre></section>`;
    return;
  }
  if (seq !== routeSeq) return;

  const selector = document.querySelector("#run-select");
  if (selector && DATA) selector.value = DATA.run_id;

  const view = VALID_VIEWS.has(route.view) ? route.view : "leaderboard";
  currentView = view === "detail" ? lastListView : view;
  if (view === "detail" && lastListView === "leaderboard") lastListView = "problems";
  if (view !== "detail") lastListView = view;
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === (view === "detail" ? lastListView : view));
  });
  app.classList.toggle("split", view === "problems" || view === "models");

  if (!DATA) {
    app.innerHTML = `<section class="card"><h2>No Runs</h2><p class="muted">No result runs were found.</p></section>`;
    return;
  }
  if (view === "leaderboard") renderLeaderboard();
  else if (view === "problems") await renderProblems(route.problemId, seq);
  else if (view === "models") renderModels(route.modelName);
  else await renderRunDetail(route.modelName, route.sampleId, seq);
}

function setView(view) {
  const route = parseHash();
  if (view === "problems") {
    navigate({
      view,
      modelName: null,
      sampleId: null,
      problemId: route.problemId || (route.view === "detail" ? route.sampleId : null),
    });
  } else if (view === "models") {
    navigate({
      view,
      problemId: null,
      sampleId: null,
      modelName: route.modelName,
    });
  } else {
    navigate({ view: "leaderboard", problemId: null, modelName: null, sampleId: null });
  }
}

function showVersionColumn() {
  return DATA.run_id === "all-versions";
}

function renderLeaderboard() {
  const tools = DATA.leaderboard.some((row) => showToolCalls(row));
  const versions = showVersionColumn();
  app.innerHTML = `<section class="card">
      <h2>Leaderboard</h2>
      <table>
        <thead><tr>
          <th>Rank</th><th>Model</th>${versions ? "<th>Version</th>" : ""}<th>Pass Rate</th><th>Passed</th>
          <th>Tokens</th>${tools ? "<th>Tools</th>" : ""}<th>Cost</th>
        </tr></thead>
        <tbody>
          ${DATA.leaderboard.map((row) => `<tr class="clickable" data-model="${esc(row.name)}">
            <td>${row.rank}</td>
            <td><strong>${esc(row.original_model || row.name)}</strong></td>
            ${versions ? `<td>${esc(runLabel(row.version || DATA.run_id))}</td>` : ""}
            <td>${pct(row.pass_rate)}</td>
            <td>${row.passed}/${row.judged}</td>
            <td>${formatCount(row.output_tokens)}</td>
            ${tools ? `<td>${showToolCalls(row) ? formatCount(row.tool_call_count) : "—"}</td>` : ""}
            <td>${money(row.total_cost_usd)}</td>
          </tr>`).join("")}
        </tbody>
      </table>
    </section>`;
  app.querySelectorAll("[data-model]").forEach((row) => {
    row.addEventListener("click", () => navigate({
      view: "models",
      modelName: row.dataset.model,
      problemId: null,
      sampleId: null,
    }));
  });
}

function bindPairButtons(root) {
  root.querySelectorAll("[data-pair]").forEach((button) => {
    const [model, sample] = button.dataset.pair.split("|||");
    button.addEventListener("click", () => navigate({ view: "detail", modelName: model, sampleId: sample }));
  });
}

async function renderProblems(selectedId = DATA.problems[0]?.id, seq = routeSeq) {
  const problem = DATA.problems.find((item) => item.id === selectedId) || DATA.problems[0];
  if (!problem) {
    app.classList.remove("split");
    app.innerHTML = `<section class="card"><h2>No Problems</h2></section>`;
    return;
  }
  const detail = problem.has_detail ? await loadProblem(problem.id) : null;
  if (seq !== routeSeq) return;
  const existing = app.querySelector(".split-layout.problem-layout");
  if (existing) {
    app.querySelectorAll("[data-problem]").forEach((button) => {
      button.classList.toggle("active", button.dataset.problem === problem.id);
    });
    const pane = app.querySelector(".detail-pane");
    pane.innerHTML = renderProblemDetail(problem, detail);
    bindPairButtons(pane);
    bindLightbox(pane);
    return;
  }
  app.innerHTML = `<div class="split-layout problem-layout">
    <aside class="card split-pane">
      <h2>Problems</h2>
      <div class="list">
        ${DATA.problems.map((item) => `<button data-problem="${esc(item.id)}" class="${item.id === problem.id ? "active" : ""}">
          <strong>Problem ${esc(item.id)}</strong><br>
          <span class="muted">${esc(item.difficulty)} · ${item.passed}/${item.judged} passed</span>
        </button>`).join("")}
      </div>
    </aside>
    <section class="split-pane detail-pane">
      ${renderProblemDetail(problem, detail)}
    </section>
  </div>`;
  app.querySelectorAll("[data-problem]").forEach((button) => {
    button.addEventListener("click", () => navigate({
      view: "problems",
      problemId: button.dataset.problem,
      modelName: null,
      sampleId: null,
    }));
  });
  bindPairButtons(app);
  bindLightbox(app);
}

function renderProblemDetail(problem, detail) {
  const published = problem.has_detail && detail && !detail.missing;
  const image = published ? detail.image : null;
  const gold = published ? detail.problem_gold_md : "";
  const solution = published ? detail.solution_text : "";
  const sourceUrl = published ? detail.source_url : "";
  const sourceLink = sourceUrl
    ? `<p class="problem-source-link"><a class="title-link" href="${esc(sourceUrl)}" target="_blank" rel="noopener noreferrer">Original problem page</a></p>`
    : "";
  const body = published
    ? `<div class="problem-source-grid">
        <div class="problem-source-panel">
          <h3>Original Image</h3>
          ${image ? `<img class="problem-image" data-zoom src="${esc(image)}" alt="Problem ${esc(problem.id)} image">` : `<p class="muted">No image found for this problem.</p>`}
          ${sourceLink}
        </div>
        <div class="problem-source-panel">
          <h3>Parsed Problem</h3>
          <div class="scroll-md">${renderMarkdown(gold)}</div>
        </div>
      </div>`
    : `<p class="muted">Problem text and image are not published for this puzzle. Scores still appear below.</p>`;
  return `<article class="card">
    <h2>Problem ${esc(problem.id)}</h2>
    <p class="muted">${esc(problem.difficulty)}</p>
    ${body}
  </article>
  ${published ? `<article class="card">
    <h3>Official Solution</h3>
    <div class="scroll-md">${renderMarkdown(solution || "No official solution text found.")}</div>
  </article>` : ""}
  <article class="card">
    <h3>Model Outcomes</h3>
    <table><thead><tr><th>Model</th><th>Status</th><th>Usage</th><th>Cost</th></tr></thead><tbody>
      ${problem.models.map((row) => renderOutcomeRow(row)).join("")}
    </tbody></table>
  </article>`;
}

function renderOutcomeRow(row) {
  const label = `${esc(row.original_model || row.model)}${row.version ? ` · ${esc(runLabel(row.version))}` : ""}`;
  if (row.status === "not_attempted") {
    return `<tr>
      <td><span class="muted">${label}</span></td>
      <td><span class="muted">Not attempted</span></td>
      <td class="muted">—</td>
      <td class="muted">—</td>
    </tr>`;
  }
  return `<tr>
    <td><button class="link-button" data-pair="${esc(row.model)}|||${esc(row.sample_id)}">${label}</button></td>
    <td>${statusPill(row.passed, row.judged)}</td>
    <td>${esc(rolloutUsage(row))}</td>
    <td>${money((row.rollout_cost_usd || 0) + (row.judge_cost_usd || 0))}</td>
  </tr>`;
}

function renderModels(selectedName = DATA.models[0]?.name) {
  const model = DATA.models.find((item) => item.name === selectedName) || DATA.models[0];
  if (!model) {
    app.classList.remove("split");
    app.innerHTML = `<section class="card"><h2>No Models</h2></section>`;
    return;
  }
  const existing = app.querySelector(".split-layout.models-layout");
  if (existing) {
    app.querySelectorAll("[data-model-list]").forEach((button) => {
      button.classList.toggle("active", button.dataset.modelList === model.name);
    });
    const detail = app.querySelector(".detail-pane");
    detail.innerHTML = `<section class="card">${renderModelTable(model)}</section>`;
    bindPairButtons(detail);
    return;
  }
  app.innerHTML = `<div class="split-layout models-layout">
    <aside class="card split-pane">
      <h2>Models</h2>
      <div class="list">
        ${DATA.models.map((item) => `<button data-model-list="${esc(item.name)}" class="${item.name === model.name ? "active" : ""}">
          <strong>${esc(item.original_model || item.name)}</strong><br>
          <span class="muted">${item.version ? `${esc(runLabel(item.version))} · ` : ""}${pct(item.pass_rate)} · ${item.passed}/${item.judged}</span>
        </button>`).join("")}
      </div>
    </aside>
    <section class="split-pane detail-pane">
      <section class="card">
        ${renderModelTable(model)}
      </section>
    </section>
  </div>`;
  app.querySelectorAll("[data-model-list]").forEach((button) => {
    button.addEventListener("click", () => navigate({
      view: "models",
      modelName: button.dataset.modelList,
      problemId: null,
      sampleId: null,
    }));
  });
  bindPairButtons(app);
}

function renderModelTable(model) {
  return `<h2>${esc(model.original_model || model.name)}</h2>
    <p class="muted">${model.version ? `${esc(runLabel(model.version))} · ` : ""}${esc(model.model_id)} · ${pct(model.pass_rate)} · ${model.passed}/${model.judged} passed</p>
    <details><summary>Model config</summary><pre>${esc(JSON.stringify(model.model_config || {}, null, 2))}</pre></details>
    <table>
      <thead><tr><th>Problem</th><th>Status</th><th>Usage</th><th>Cost</th></tr></thead>
      <tbody>${model.problems.map((row) => `<tr>
        <td><button class="link-button" data-pair="${esc(model.name)}|||${esc(row.sample_id)}">Problem ${esc(row.sample_id)}</button></td>
        <td>${statusPill(row.passed, row.judged)}</td>
        <td>${esc(rolloutUsage({...row, version: row.version || model.version}))}</td>
        <td>${money((row.rollout_cost_usd || 0) + (row.judge_cost_usd || 0))}</td>
      </tr>`).join("")}</tbody>
    </table>`;
}

async function renderRunDetail(modelName, sampleId, seq = routeSeq) {
  const detail = await loadDetail(modelName, sampleId);
  if (seq !== routeSeq) return;
  if (!detail) {
    navigate({ view: lastListView === "models" ? "models" : "problems", modelName, sampleId: null, problemId: sampleId }, { replace: true });
    return;
  }
  app.classList.remove("split");
  app.innerHTML = `<button class="link-button" id="back-button">Back to ${esc(lastListView)}</button>
    <section class="grid">
      <article class="card">
        <h2>${esc(detail.original_model || modelName)} · Problem ${esc(sampleId)}</h2>
        <p>${statusPill(detail.passed, detail.judged)} <span class="muted">${esc(rolloutUsage({
          ...detail,
          version: detail.version || DATA.run_id,
        }))}</span></p>
      </article>
      <article class="card">
        <h3>Model Proposed Solution</h3>
        <pre>${esc(detail.model_solution || "No final solution found.")}</pre>
      </article>
      ${renderJudge(detail)}
    </section>`;
  document.querySelector("#back-button").addEventListener("click", () => {
    if (lastListView === "models") {
      navigate({ view: "models", modelName, problemId: null, sampleId: null });
    } else {
      navigate({ view: "problems", problemId: sampleId, modelName: null, sampleId: null });
    }
  });
}

function renderJudge(detail) {
  if (!detail.judged) {
    return `<article class="card"><h3>Judge Verdict</h3><p class="muted">No judge file found yet.</p></article>`;
  }
  return `<article class="card">
    <h3>Judge Verdict</h3>
    <p>${statusPill(detail.passed, true)}</p>
    <pre>${esc(detail.judge_reasoning || "")}</pre>
  </article>`;
}

const lightbox = document.querySelector("#lightbox");
if (lightbox) {
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox || event.target.classList.contains("lightbox-close")) closeLightbox();
  });
}
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeLightbox();
});

fetch("data/manifest.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error(`data/manifest.json (${response.status})`);
    return response.json();
  })
  .then((data) => {
    MANIFEST = data;
    initRunSelector();
    document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
    window.addEventListener("hashchange", () => applyRoute());
    if (!location.hash) {
      history.replaceState(null, "", routeHash(defaultRoute()));
    }
    applyRoute();
  })
  .catch((error) => {
    app.innerHTML = `<section class="card"><h2>Could not load data/manifest.json</h2><pre>${esc(error.stack || error.message || error)}</pre></section>`;
  });
