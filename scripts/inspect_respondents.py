#!/usr/bin/env python3
"""
inspect_respondents.py
----------------------
Generates output/inspect_respondents.html — a two-tab inspector for
reviewing stimuli before fieldwork.

  Tab 1  Respondent   — navigate by ID; see the 6 vignettes assigned in order
  Tab 2  Condition    — filter by topic / severity / ideology / age group;
                        browse every unique stimulus that matches

Open the output file from the project root so that relative iframe paths
(html/stimulus_XXXXX.html) resolve correctly.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

METADATA = Path("output/metadata/vignette_metadata.csv")
OUTPUT   = Path("output/inspect_respondents.html")

TOPIC_COLOR = {
    "Racism / ethnicity":                   "#fa709a",
    "Religion (Muslim / Jewish)":           "#4facfe",
    "Immigration / migrants":               "#667eea",
    "Gender issues (misogyny)":             "#43e97b",
    "Sexual orientation / gender identity": "#f093fb",
    "Nationalism / identity politics":      "#a18cd1",
}
TOPIC_EMOJI = {
    "Racism / ethnicity":                   "✊🏾",
    "Religion (Muslim / Jewish)":           "🕌",
    "Immigration / migrants":               "🌍",
    "Gender issues (misogyny)":             "✊",
    "Sexual orientation / gender identity": "🏳️‍🌈",
    "Nationalism / identity politics":      "🏴",
}
SEV_COLOR   = {"opinion": "#3b82f6", "dehumanising": "#f59e0b", "incitement": "#ef4444"}
IDEO_COLOR  = {"conservative": "#6366f1", "progressive": "#0d9488"}
AGE_COLOR   = {"adolescent": "#f97316", "young_adult": "#8b5cf6"}

TOPICS     = list(TOPIC_COLOR)
SEVERITIES = ["opinion", "dehumanising", "incitement"]
IDEOLOGIES = ["conservative", "progressive"]
AGE_GROUPS = ["adolescent", "young_adult"]


# ── data ─────────────────────────────────────────────────────────────────────

def read_metadata():
    with open(METADATA, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_data(rows):
    by_resp = defaultdict(list)
    for r in rows:
        by_resp[int(r["respondent_id"])].append(r)

    respondents = []
    for resp_id in sorted(by_resp):
        vigs = sorted(by_resp[resp_id], key=lambda x: int(x["vignette_order"]))
        respondents.append({
            "id": resp_id,
            "v": [{
                "o": int(v["vignette_order"]),
                "t": v["topic"],
                "s": v["severity"],
                "i": v["ideology"],
                "a": v["age_group"],
                "f": v["stimulus_filename"].replace(".png", ".html"),
                "c": v["comment_text"],
                "p": v["profile_id"],
            } for v in vigs],
        })

    # One entry per unique stimulus for the condition browser
    seen: dict = {}
    for r in rows:
        fn = r["stimulus_filename"].replace(".png", ".html")
        if fn not in seen:
            seen[fn] = {
                "f": fn,
                "t": r["topic"],
                "s": r["severity"],
                "i": r["ideology"],
                "a": r["age_group"],
                "c": r["comment_text"],
                "p": r["profile_id"],
            }
    stimuli = list(seen.values())

    return respondents, stimuli


# ── html ──────────────────────────────────────────────────────────────────────

# Use plain string (not f-string) so JS braces don't need escaping.
# Data is injected via .replace() on sentinel tokens.

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stimulus Inspector</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f1f5f9; color: #1e293b; font-size: 13px;
}

/* ── header ── */
.page-header {
  background: #1e293b; color: #f8fafc;
  padding: 14px 24px; display: flex; align-items: baseline; gap: 14px;
}
.page-header h1 { font-size: 17px; font-weight: 700; }
.page-header .sub { font-size: 12px; color: #94a3b8; }

/* ── tabs ── */
.tabs {
  background: #fff; border-bottom: 1px solid #e2e8f0;
  display: flex; padding: 0 20px; gap: 4px;
}
.tab-btn {
  background: none; border: none; padding: 12px 18px;
  font-size: 13px; font-weight: 600; color: #64748b;
  cursor: pointer; border-bottom: 3px solid transparent; margin-bottom: -1px;
}
.tab-btn.active { color: #3b82f6; border-bottom-color: #3b82f6; }

/* ── panels ── */
.panel { display: none; padding: 20px 24px; }
.panel.active { display: block; }

/* ── respondent nav ── */
.resp-nav {
  display: flex; align-items: center; gap: 10px;
  background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
  padding: 12px 16px; margin-bottom: 16px; flex-wrap: wrap;
}
.resp-nav label { font-weight: 600; font-size: 13px; color: #475569; }
.resp-nav input[type=number] {
  width: 80px; padding: 5px 8px; border: 1px solid #cbd5e1;
  border-radius: 6px; font-size: 13px; font-weight: 700; text-align: center;
}
.resp-nav .total { color: #94a3b8; font-size: 12px; }
.nav-btn {
  background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px;
  padding: 5px 12px; cursor: pointer; font-size: 13px; font-weight: 600;
  color: #475569;
}
.nav-btn:hover { background: #e2e8f0; }
.resp-summary {
  display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto;
}

/* ── vignette row ── */
.vignette-row {
  display: flex; gap: 14px; overflow-x: auto; padding-bottom: 8px;
}
.vig-card {
  flex-shrink: 0; background: #fff; border: 1px solid #e2e8f0;
  border-radius: 10px; overflow: hidden; width: 192px;
}
.vig-card-header {
  padding: 8px 10px 6px; border-bottom: 1px solid #f1f5f9;
  display: flex; flex-direction: column; gap: 4px;
}
.vig-order { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; }
.badges { display: flex; flex-wrap: wrap; gap: 3px; }
.badge {
  border-radius: 999px; padding: 2px 7px; font-size: 10px; font-weight: 700;
  white-space: nowrap;
}
.vig-profile { font-size: 10px; color: #94a3b8; padding: 0 10px 4px; }

/* ── scaled iframe ── */
.frame-wrap {
  width: 192px; height: 355px; overflow: hidden; position: relative;
  background: #f8fafc;
}
.frame-wrap iframe {
  width: 375px; height: 710px;
  transform: scale(0.512); transform-origin: 0 0;
  border: none; pointer-events: none;
}

/* ── comment preview ── */
.vig-comment {
  padding: 8px 10px; font-size: 11px; line-height: 1.5; color: #475569;
  border-top: 1px solid #f1f5f9; background: #fafafa;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
  overflow: hidden;
}

/* ── condition filters ── */
.filter-bar {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
  padding: 12px 16px; margin-bottom: 16px;
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
}
.filter-bar label { font-weight: 600; font-size: 12px; color: #64748b; }
.filter-bar select {
  padding: 5px 10px; border: 1px solid #cbd5e1; border-radius: 6px;
  font-size: 12px; background: #f8fafc; color: #1e293b; cursor: pointer;
}
.filter-bar select:focus { outline: none; border-color: #3b82f6; }
.result-count {
  margin-left: auto; font-size: 12px; color: #64748b; font-weight: 600;
}
.reset-btn {
  background: none; border: 1px solid #e2e8f0; border-radius: 6px;
  padding: 5px 10px; font-size: 12px; cursor: pointer; color: #64748b;
}
.reset-btn:hover { background: #f1f5f9; }

/* ── stimulus grid ── */
.stim-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, 192px);
  gap: 14px;
}
.stim-card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
  overflow: hidden;
}
.stim-card-header {
  padding: 8px 10px; border-bottom: 1px solid #f1f5f9;
  display: flex; flex-direction: column; gap: 4px;
}
.stim-profile { font-size: 10px; color: #94a3b8; }
.no-results {
  grid-column: 1/-1; padding: 40px; text-align: center; color: #94a3b8;
  font-size: 14px;
}
</style>
</head>
<body>

<div class="page-header">
  <h1>Stimulus Inspector</h1>
  <span class="sub">__N_RESP__ respondents · __N_STIM__ unique stimuli</span>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('respondent', this)">
    Respondent View
  </button>
  <button class="tab-btn" onclick="showTab('condition', this)">
    Condition Browser
  </button>
</div>

<!-- ── Tab 1: Respondent View ───────────────────────────────────────────── -->
<div id="panel-respondent" class="panel active">

  <div class="resp-nav">
    <label>Respondent</label>
    <button class="nav-btn" onclick="navigate(-1)">&#8592; Prev</button>
    <input type="number" id="resp-input" min="1" max="__N_RESP__" value="1"
           onchange="jumpTo(this.value)">
    <span class="total">of __N_RESP__</span>
    <button class="nav-btn" onclick="navigate(1)">Next &#8594;</button>
    <div class="resp-summary" id="resp-summary"></div>
  </div>

  <div class="vignette-row" id="vig-row"></div>
</div>

<!-- ── Tab 2: Condition Browser ─────────────────────────────────────────── -->
<div id="panel-condition" class="panel">

  <div class="filter-bar">
    <label>Topic</label>
    <select id="f-topic" onchange="renderGrid()">
      <option value="">All topics</option>
      __TOPIC_OPTIONS__
    </select>

    <label>Severity</label>
    <select id="f-severity" onchange="renderGrid()">
      <option value="">All</option>
      <option>opinion</option>
      <option>dehumanising</option>
      <option>incitement</option>
    </select>

    <label>Ideology</label>
    <select id="f-ideology" onchange="renderGrid()">
      <option value="">All</option>
      <option>conservative</option>
      <option>progressive</option>
    </select>

    <label>Age group</label>
    <select id="f-age" onchange="renderGrid()">
      <option value="">All</option>
      <option>adolescent</option>
      <option>young_adult</option>
    </select>

    <span class="result-count" id="result-count"></span>
    <button class="reset-btn" onclick="resetFilters()">Reset</button>
  </div>

  <div class="stim-grid" id="stim-grid"></div>
</div>

<script>
// ── data ────────────────────────────────────────────────────────────────────
const RESPONDENTS = __RESPONDENTS__;
const STIMULI     = __STIMULI__;

// ── colour helpers ───────────────────────────────────────────────────────────
const TOPIC_COLOR = {
  "Racism / ethnicity":                   "#fa709a",
  "Religion (Muslim / Jewish)":           "#4facfe",
  "Immigration / migrants":               "#667eea",
  "Gender issues (misogyny)":             "#43e97b",
  "Sexual orientation / gender identity": "#f093fb",
  "Nationalism / identity politics":      "#a18cd1",
};
const TOPIC_EMOJI = {
  "Racism / ethnicity":                   "✊🏾",
  "Religion (Muslim / Jewish)":           "🕌",
  "Immigration / migrants":               "🌍",
  "Gender issues (misogyny)":             "✊",
  "Sexual orientation / gender identity": "🏳️‍🌈",
  "Nationalism / identity politics":      "🏴",
};
const SEV_COLOR  = { opinion:"#3b82f6", dehumanising:"#f59e0b", incitement:"#ef4444" };
const IDEO_COLOR = { conservative:"#6366f1", progressive:"#0d9488" };
const AGE_COLOR  = { adolescent:"#f97316", young_adult:"#8b5cf6" };

function badge(text, color) {
  return `<span class="badge" style="background:${color}22;color:${color};border:1px solid ${color}44">${text}</span>`;
}
function topicBadge(t) {
  const c = TOPIC_COLOR[t] || "#888";
  return badge((TOPIC_EMOJI[t] || "") + " " + t.split(" /")[0].split("(")[0].trim(), c);
}
function sevBadge(s)  { return badge(s, SEV_COLOR[s] || "#888"); }
function ideoBadge(i) { return badge(i, IDEO_COLOR[i] || "#888"); }
function ageBadge(a)  { return badge(a.replace("_", " "), AGE_COLOR[a] || "#888"); }

function frameHTML(filename) {
  return `<div class="frame-wrap">
    <iframe src="html/${filename}" loading="lazy" title="${filename}"></iframe>
  </div>`;
}

// ── Tab 1: Respondent view ───────────────────────────────────────────────────
let currentIdx = 0;

function renderRespondent(idx) {
  const r = RESPONDENTS[idx];
  if (!r) return;
  currentIdx = idx;
  document.getElementById("resp-input").value = r.id;

  // Summary badges in nav bar
  const summaryEl = document.getElementById("resp-summary");
  summaryEl.innerHTML = r.v.map(v =>
    `<span title="${v.t} · ${v.s} · ${v.i} · ${v.a}"
           style="display:inline-block;width:10px;height:10px;border-radius:50%;
                  background:${TOPIC_COLOR[v.t] || '#888'};
                  border:2px solid ${SEV_COLOR[v.s] || '#888'}"
     ></span>`
  ).join("");

  // Vignette cards
  const rowEl = document.getElementById("vig-row");
  rowEl.innerHTML = r.v.map(v => `
    <div class="vig-card">
      <div class="vig-card-header">
        <div class="vig-order">Vignette ${v.o} of 6</div>
        <div class="badges">
          ${topicBadge(v.t)}
        </div>
        <div class="badges">
          ${sevBadge(v.s)} ${ideoBadge(v.i)} ${ageBadge(v.a)}
        </div>
      </div>
      <div class="vig-profile">${v.p}</div>
      ${frameHTML(v.f)}
      <div class="vig-comment">${v.c}</div>
    </div>
  `).join("");
}

function navigate(delta) {
  const next = Math.max(0, Math.min(RESPONDENTS.length - 1, currentIdx + delta));
  renderRespondent(next);
}

function jumpTo(val) {
  const id = parseInt(val, 10);
  if (isNaN(id)) return;
  const idx = RESPONDENTS.findIndex(r => r.id === id);
  if (idx >= 0) renderRespondent(idx);
}

// ── Tab 2: Condition browser ─────────────────────────────────────────────────
function getFilters() {
  return {
    topic:    document.getElementById("f-topic").value,
    severity: document.getElementById("f-severity").value,
    ideology: document.getElementById("f-ideology").value,
    age:      document.getElementById("f-age").value,
  };
}

function renderGrid() {
  const f = getFilters();
  const filtered = STIMULI.filter(s =>
    (!f.topic    || s.t === f.topic)    &&
    (!f.severity || s.s === f.severity) &&
    (!f.ideology || s.i === f.ideology) &&
    (!f.age      || s.a === f.age)
  );

  document.getElementById("result-count").textContent =
    `${filtered.length.toLocaleString()} stimuli`;

  const grid = document.getElementById("stim-grid");
  if (filtered.length === 0) {
    grid.innerHTML = `<div class="no-results">No stimuli match the selected filters.</div>`;
    return;
  }

  grid.innerHTML = filtered.map(s => `
    <div class="stim-card">
      <div class="stim-card-header">
        <div class="badges">${topicBadge(s.t)}</div>
        <div class="badges">${sevBadge(s.s)} ${ideoBadge(s.i)} ${ageBadge(s.a)}</div>
        <div class="stim-profile">${s.p}</div>
      </div>
      ${frameHTML(s.f)}
      <div class="vig-comment">${s.c}</div>
    </div>
  `).join("");
}

function resetFilters() {
  ["f-topic","f-severity","f-ideology","f-age"].forEach(id => {
    document.getElementById(id).value = "";
  });
  renderGrid();
}

// ── tabs ─────────────────────────────────────────────────────────────────────
function showTab(name, btn) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("panel-" + name).classList.add("active");
  btn.classList.add("active");
}

// ── init ─────────────────────────────────────────────────────────────────────
renderRespondent(0);
renderGrid();
</script>
</body>
</html>"""


def make_topic_options():
    return "\n      ".join(
        f'<option value="{t}">{t}</option>' for t in TOPICS
    )


def build_html(respondents: list, stimuli: list) -> str:
    return (
        HTML_TEMPLATE
        .replace("__N_RESP__",        str(len(respondents)))
        .replace("__N_STIM__",        str(len(stimuli)))
        .replace("__RESPONDENTS__",   json.dumps(respondents, ensure_ascii=False))
        .replace("__STIMULI__",       json.dumps(stimuli,     ensure_ascii=False))
        .replace("__TOPIC_OPTIONS__",  make_topic_options())
    )


def main() -> None:
    rows = read_metadata()
    respondents, stimuli = build_data(rows)
    page = build_html(respondents, stimuli)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page, encoding="utf-8")
    size_mb = OUTPUT.stat().st_size / 1_048_576
    print(f"Written → {OUTPUT}  ({size_mb:.1f} MB)")
    print(f"  {len(respondents):,} respondents  ·  {len(stimuli):,} unique stimuli")


if __name__ == "__main__":
    main()
