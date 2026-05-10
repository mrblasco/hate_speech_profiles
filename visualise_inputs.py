#!/usr/bin/env python3
"""
visualise_inputs.py
-------------------
Generates output/visualise_inputs.html — a self-contained dashboard that
lets you review all three input CSVs before running the full pipeline.

  Tab 1  Comments  — 6-topic grid, severity rows × ideology columns
  Tab 2  Profiles  — profile cards grouped by topic
  Tab 3  Engagement — simple table of the 9 engagement rows

No third-party dependencies (stdlib only).
"""

import csv
import html as _html
from pathlib import Path

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")

# ── severity display config ───────────────────────────────────────────────────
SEVERITY_ORDER = ["opinion", "dehumanising", "incitement"]
SEVERITY_META  = {
    "opinion":      {"label": "Opinion",      "color": "#3b82f6", "bg": "#eff6ff"},
    "dehumanising": {"label": "Dehumanising", "color": "#f59e0b", "bg": "#fffbeb"},
    "incitement":   {"label": "Incitement",   "color": "#ef4444", "bg": "#fef2f2"},
}

# ── topic display config (gradient approximated as solid for badges) ───────────
TOPIC_ORDER = [
    "Racism / ethnicity",
    "Religion (Muslim / Jewish)",
    "Immigration / migrants",
    "Gender issues (misogyny)",
    "Sexual orientation / gender identity",
    "Nationalism / identity politics",
]
TOPIC_META = {
    "Racism / ethnicity":                  {"emoji": "✊🏾", "color": "#fa709a"},
    "Religion (Muslim / Jewish)":          {"emoji": "🕌",  "color": "#4facfe"},
    "Immigration / migrants":              {"emoji": "🌍",  "color": "#667eea"},
    "Gender issues (misogyny)":            {"emoji": "✊",   "color": "#43e97b"},
    "Sexual orientation / gender identity":{"emoji": "🏳️‍🌈","color": "#f093fb"},
    "Nationalism / identity politics":     {"emoji": "🏴",  "color": "#a18cd1"},
}

IDEOLOGY_META = {
    "conservative": {"label": "Conservative", "color": "#6366f1", "bg": "#eef2ff"},
    "progressive":  {"label": "Progressive",  "color": "#0d9488", "bg": "#f0fdfa"},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def e(text: str) -> str:
    return _html.escape(str(text))


def read_csv(name: str) -> list[dict]:
    with open(DATA_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── section builders ──────────────────────────────────────────────────────────

def build_comments_section(rows: list[dict]) -> str:
    # index: topic → severity → ideology → [texts]
    index: dict = {}
    for r in rows:
        t, s, i, txt = r["topic"], r["severity"], r["ideology"], r["text"]
        index.setdefault(t, {}).setdefault(s, {}).setdefault(i, []).append(txt)

    parts = []
    for topic in TOPIC_ORDER:
        tm = TOPIC_META[topic]
        safe_id = topic.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")

        parts.append(f"""
        <div class="topic-block">
          <div class="topic-header" style="border-left: 4px solid {e(tm['color'])};">
            <span class="topic-emoji">{tm['emoji']}</span>
            <span class="topic-name">{e(topic)}</span>
          </div>
          <div class="comment-grid">
            <div class="grid-corner"></div>
            <div class="col-head" style="background:{e(IDEOLOGY_META['conservative']['bg'])};
                 color:{e(IDEOLOGY_META['conservative']['color'])};">Conservative</div>
            <div class="col-head" style="background:{e(IDEOLOGY_META['progressive']['bg'])};
                 color:{e(IDEOLOGY_META['progressive']['color'])};">Progressive</div>
        """)

        for sev in SEVERITY_ORDER:
            sm = SEVERITY_META[sev]
            parts.append(f"""
            <div class="row-head" style="background:{e(sm['bg'])};color:{e(sm['color'])};">
              {e(sm['label'])}
            </div>
            """)
            for ideo in ["conservative", "progressive"]:
                texts = index.get(topic, {}).get(sev, {}).get(ideo, [])
                im = IDEOLOGY_META[ideo]
                parts.append(f'<div class="cell" style="border-top:2px solid {e(sm["color"])}30;">')
                for txt in texts:
                    parts.append(f'<p class="comment-item">{e(txt)}</p>')
                parts.append("</div>")

        parts.append("</div></div>")  # close grid + topic-block

    return "\n".join(parts)


def build_profiles_section(rows: list[dict]) -> str:
    by_topic: dict = {}
    for r in rows:
        by_topic.setdefault(r["topic"], []).append(r)

    parts = []
    for topic in TOPIC_ORDER:
        if topic not in by_topic:
            continue
        tm = TOPIC_META[topic]
        parts.append(f"""
        <div class="topic-block">
          <div class="topic-header" style="border-left:4px solid {e(tm['color'])};">
            <span class="topic-emoji">{tm['emoji']}</span>
            <span class="topic-name">{e(topic)}</span>
          </div>
          <div class="card-grid">
        """)
        for p in sorted(by_topic[topic], key=lambda x: x["profile_id"]):
            age_badge = ("🧑 Young adult" if p["age_group"] == "young_adult"
                         else "🧔 Middle adult")
            parts.append(f"""
            <div class="profile-card">
              <div class="profile-card-top">
                <div class="avatar-circle" style="background:{e(p['avatar_colour'])};">
                  {e(p['avatar_initials'])}
                </div>
                <div class="profile-info">
                  <div class="profile-id">{e(p['profile_id'])}</div>
                  <div class="profile-name">{e(p['display_name'])}</div>
                  <div class="profile-user">@{e(p['username'])}</div>
                </div>
              </div>
              <div class="profile-meta">
                <span class="meta-badge">{e(age_badge)}</span>
                <span class="meta-badge">Age {e(p['age'])}</span>
                <span class="meta-badge">{e(p['gender'].title())}</span>
              </div>
              <div class="profile-origin">{e(p['origin'])}</div>
              <p class="profile-msg">{e(p['target_message'])}</p>
            </div>
            """)
        parts.append("</div></div>")

    return "\n".join(parts)


def build_engagement_section(rows: list[dict]) -> str:
    level_order = ["low", "medium", "high"]
    level_color = {"low": "#6b7280", "medium": "#3b82f6", "high": "#ef4444"}

    by_level: dict = {}
    for r in rows:
        by_level.setdefault(r["engagement_level"], []).append(r)

    parts = ['<div class="eng-wrapper"><table class="eng-table">',
             "<thead><tr><th>Level</th><th>Likes</th><th>Comments</th></tr></thead>",
             "<tbody>"]
    for lvl in level_order:
        color = level_color[lvl]
        for r in by_level.get(lvl, []):
            parts.append(f"""
            <tr>
              <td><span class="eng-badge" style="background:{color}20;color:{color};
                   border:1px solid {color}50;">{e(lvl.title())}</span></td>
              <td class="eng-num">❤️ {e(r['likes'])}</td>
              <td class="eng-num">💬 {e(r['comments_count'])}</td>
            </tr>
            """)
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


# ── full page ─────────────────────────────────────────────────────────────────

def build_page(comments: list[dict], profiles: list[dict], engagement: list[dict]) -> str:
    comments_html   = build_comments_section(comments)
    profiles_html   = build_profiles_section(profiles)
    engagement_html = build_engagement_section(engagement)

    n_comments  = len(comments)
    n_profiles  = len(profiles)
    n_eng       = len(engagement)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Input Data Viewer — Hate Speech Vignette Study</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  font-size: 14px;
  line-height: 1.5;
}}

/* ── header ── */
.page-header {{
  background: #1e293b;
  color: #f8fafc;
  padding: 18px 28px;
  display: flex;
  align-items: baseline;
  gap: 16px;
}}
.page-header h1 {{ font-size: 18px; font-weight: 700; }}
.page-header .subtitle {{ font-size: 13px; color: #94a3b8; }}

/* ── tabs ── */
.tabs {{
  display: flex;
  background: #fff;
  border-bottom: 1px solid #e2e8f0;
  padding: 0 20px;
  gap: 4px;
}}
.tab-btn {{
  background: none;
  border: none;
  padding: 14px 20px;
  font-size: 14px;
  font-weight: 600;
  color: #64748b;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  margin-bottom: -1px;
}}
.tab-btn.active {{
  color: #3b82f6;
  border-bottom-color: #3b82f6;
}}
.tab-btn:hover:not(.active) {{ color: #1e293b; }}
.tab-count {{
  display: inline-block;
  background: #e2e8f0;
  border-radius: 999px;
  padding: 1px 7px;
  font-size: 11px;
  font-weight: 700;
  margin-left: 6px;
  color: #64748b;
}}

/* ── tab panels ── */
.tab-panel {{ display: none; padding: 24px 20px; max-width: 1200px; margin: 0 auto; }}
.tab-panel.active {{ display: block; }}

/* ── topic blocks ── */
.topic-block {{ margin-bottom: 40px; }}
.topic-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: #fff;
  border-radius: 8px 8px 0 0;
  border: 1px solid #e2e8f0;
  margin-bottom: 0;
}}
.topic-emoji {{ font-size: 20px; }}
.topic-name {{ font-size: 15px; font-weight: 700; color: #1e293b; }}

/* ── comment grid ── */
.comment-grid {{
  display: grid;
  grid-template-columns: 100px 1fr 1fr;
  border: 1px solid #e2e8f0;
  border-top: none;
  border-radius: 0 0 8px 8px;
  overflow: hidden;
  background: #fff;
}}
.grid-corner {{
  background: #f1f5f9;
  border-right: 1px solid #e2e8f0;
  border-bottom: 1px solid #e2e8f0;
}}
.col-head {{
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 10px 14px;
  border-bottom: 1px solid #e2e8f0;
  border-right: 1px solid #e2e8f0;
  text-align: center;
}}
.col-head:last-child {{ border-right: none; }}
.row-head {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  padding: 12px 10px;
  border-right: 1px solid #e2e8f0;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  writing-mode: vertical-rl;
  transform: rotate(180deg);
}}
.cell {{
  padding: 12px 14px;
  border-right: 1px solid #e2e8f0;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.cell:last-child {{ border-right: none; }}
.comment-item {{
  background: #f8fafc;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 13px;
  line-height: 1.5;
  color: #334155;
  border: 1px solid #e2e8f0;
}}

/* ── profile cards ── */
.card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
  margin-top: 2px;
  padding: 12px;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-top: none;
  border-radius: 0 0 8px 8px;
}}
.profile-card {{
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.profile-card-top {{ display: flex; gap: 12px; align-items: flex-start; }}
.avatar-circle {{
  width: 44px;
  height: 44px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 14px;
  color: #fff;
  flex-shrink: 0;
}}
.profile-info {{ display: flex; flex-direction: column; gap: 2px; }}
.profile-id {{ font-size: 11px; font-weight: 600; color: #94a3b8; }}
.profile-name {{ font-size: 14px; font-weight: 700; color: #1e293b; }}
.profile-user {{ font-size: 12px; color: #64748b; }}
.profile-meta {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.meta-badge {{
  background: #e2e8f0;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  color: #475569;
}}
.profile-origin {{ font-size: 12px; font-style: italic; color: #64748b; }}
.profile-msg {{
  font-size: 12px;
  color: #475569;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 8px;
  line-height: 1.5;
}}

/* ── engagement table ── */
.eng-wrapper {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; max-width: 500px; }}
.eng-table {{ width: 100%; border-collapse: collapse; }}
.eng-table thead {{ background: #f1f5f9; }}
.eng-table th {{
  padding: 12px 16px;
  text-align: left;
  font-size: 12px;
  font-weight: 700;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 1px solid #e2e8f0;
}}
.eng-table td {{
  padding: 12px 16px;
  border-bottom: 1px solid #f1f5f9;
  vertical-align: middle;
}}
.eng-table tr:last-child td {{ border-bottom: none; }}
.eng-badge {{
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 12px;
  font-weight: 600;
}}
.eng-num {{ font-variant-numeric: tabular-nums; font-size: 13px; }}
</style>
</head>
<body>

<div class="page-header">
  <h1>Input Data Viewer</h1>
  <span class="subtitle">Hate Speech Vignette Study — 6 topics · 3 severities · 2 ideologies</span>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('comments', this)">
    Hate Comments <span class="tab-count">{n_comments}</span>
  </button>
  <button class="tab-btn" onclick="showTab('profiles', this)">
    Profiles <span class="tab-count">{n_profiles}</span>
  </button>
  <button class="tab-btn" onclick="showTab('engagement', this)">
    Engagement <span class="tab-count">{n_eng}</span>
  </button>
</div>

<div id="tab-comments" class="tab-panel active">
  {comments_html}
</div>

<div id="tab-profiles" class="tab-panel">
  {profiles_html}
</div>

<div id="tab-engagement" class="tab-panel">
  <p style="color:#64748b;margin-bottom:16px;font-size:13px;">
    Three rows per engagement level; one is sampled at random per vignette.
  </p>
  {engagement_html}
</div>

<script>
function showTab(name, btn) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>"""


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comments   = read_csv("hate_comments.csv")
    profiles   = read_csv("profiles.csv")
    engagement = read_csv("engagement.csv")

    page = build_page(comments, profiles, engagement)

    out = OUTPUT_DIR / "visualise_inputs.html"
    out.write_text(page, encoding="utf-8")
    print(f"Written → {out}  ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
