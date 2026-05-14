"""
HTML (and optional PNG) stimulus renderer.

Converts final StimulusRow objects into Instagram-style HTML files using
templates/instagram_post.html + static/style.css, then optionally fires
Playwright to produce HiDPI PNG screenshots.

All visual decoration choices (avatar colour, commenter username, engagement
numbers, post age) are derived deterministically from the run seed so the
visual output is fully reproducible.
"""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

from src.models import StimulusRow
from src.utils.logging_utils import get_logger
from src.utils.seeds import derive_seed

# ── Load stimulus factors from config ─────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "study_config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _cfg_data = yaml.safe_load(_f)
    _design   = _cfg_data["design"]

STIMULUS_FACTORS: dict[str, list[str]] = {
    k: _design.get(k, [])
    for k in _cfg_data.get("stimulus_factors", [])
}

# Likes and comment counts per engagement level (used when post_engagement is a factor)
ENGAGEMENT_COUNTS: dict[str, tuple[int, int]] = {
    "low":  (15, 3),
    "high": (4821, 187),
}

log = get_logger("generators.html")

# ── Visual constants ──────────────────────────────────────────────────────────

AVATAR_COLOURS = [
    "#E91E8C", "#2196F3", "#9C27B0", "#FF5722",
    "#4CAF50", "#FF9800", "#00BCD4", "#F44336",
    "#3F51B5", "#009688", "#8BC34A", "#FFC107",
]

TOPIC_GRADIENTS: dict[str, str] = {
    "immigration":       "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "feminism":          "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
    "religion":          "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    "climate":           "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
    "public_health":     "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    "national_identity": "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
}

TOPIC_EMOJIS: dict[str, str] = {
    "immigration":       "🌍",
    "feminism":          "✊",
    "religion":          "🕌",
    "climate":           "🌱",
    "public_health":     "💉",
    "national_identity": "🏴",
}

ANONYMOUS_USERNAME = "anonymous_user"

COMMENTER_USERNAMES = [
    "marco.rossi91",    "giulia.ferrari",   "luca.bianchi",    "sofia.esposito",
    "matteo.romano",    "francesca.ricci",  "andrea.conti",    "elena.colombo",
    "davide.greco",     "chiara.marino",    "simone.bruno",    "valentina.gallo",
    "federico.costa",   "alessia.fontana",  "giacomo.moretti", "sara.barbieri",
    "lorenzo.riva",     "martina.silvestri","emanuele.caruso", "laura.ferretti",
    "thomas.mueller",   "anna.schmidt",     "pierre.dubois",   "marie.lefevre",
    "carlos.garcia",    "isabelle.martin",  "jan.kowalski",    "katarzyna.nowak",
]

POST_AGES = [
    "2 HOURS AGO", "3 HOURS AGO", "4 HOURS AGO",
    "5 HOURS AGO", "6 HOURS AGO", "8 HOURS AGO",
]


_FALLBACK_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"

CAPTION_PREVIEW_LENGTH = 120
SCREENSHOT_BATCH_SIZE = 20
SCREENSHOT_VIEWPORT = {"width": 375, "height": 812}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick(pool: list, seed_int: int):
    return pool[seed_int % len(pool)]


def _avatar_initials(display_name: str) -> str:
    return "".join(w[0].upper() for w in display_name.split()[:2])


# ── Context builder ───────────────────────────────────────────────────────────

def _row_decoration(row: StimulusRow, base_seed: int, anonymous: bool = False) -> tuple[str, str, str]:
    """Return (avatar_colour, commenter_username, post_age) — stable across display variants."""
    avatar_colour = _pick(AVATAR_COLOURS, derive_seed(base_seed, "avatar", row.profile_id))
    post_age      = _pick(POST_AGES,      derive_seed(base_seed, "age",    row.stimulus_id))
    if anonymous:
        anon_num = derive_seed(base_seed, "anon_id", row.comment_id) % 90000 + 10000
        commenter_username = f"user_{anon_num}"
    else:
        commenter_username = _pick(COMMENTER_USERNAMES, derive_seed(base_seed, "commenter", row.comment_id))
    return avatar_colour, commenter_username, post_age


def build_context(
    row: StimulusRow,
    css: str,
    avatar_colour: str,
    commenter_username: str,
    post_age: str,
    likes: int,
    comments_count: int,
) -> dict:
    topic = row.topic

    n = f"{comments_count:,}"
    likes_fmt = f"{likes:,}"

    caption = row.caption
    caption_truncated = len(caption) > CAPTION_PREVIEW_LENGTH
    caption_preview = caption[:CAPTION_PREVIEW_LENGTH].rstrip() + "…" if caption_truncated else caption

    return {
        "css":                css,
        "lang":               "en",
        "username":           row.username,
        "display_name":       row.display_name,
        "avatar_initials":    _avatar_initials(row.display_name),
        "avatar_colour":      avatar_colour,
        "target_message":     row.caption,
        "topic_gradient":     TOPIC_GRADIENTS.get(topic, _FALLBACK_GRADIENT),
        "topic_emoji":        TOPIC_EMOJIS.get(topic, "📱"),
        "commenter_username": commenter_username,
        "comment_text":       row.comment_text,
        "likes":              likes_fmt,
        "comments_count":     n,
        "post_age":           post_age,
        "ui_likes":           f"{likes_fmt} likes",
        "ui_view_all":        f"View all {n} comments",
        "ui_view":            f"View {n} comments",
        "ui_more":            "… more",
        "caption_preview":    caption_preview,
        "caption_truncated":  caption_truncated,
    }


# ── HTML generation ───────────────────────────────────────────────────────────

def generate_html_stimuli(
    rows: list[StimulusRow],
    output_dir: Path,
    project_root: Path,
    base_seed: int,
) -> list[Path]:
    """
    Render one HTML file per StimulusRow × stimulus factor combination.

    With the default 3×2×2 design:
      - 3 StimulusRows per post  (one per comment severity, from generation)
      - × 2 anonymity levels     (stimulus_factors: comment_anonymity)
      - × 2 engagement levels    (stimulus_factors: post_engagement)
      = 12 HTML files per topic

    Filename: {stimulus_id}_{factor_value_slugs}.html
    e.g. P0001_POST00_OPP00_anonymous_low.html

    CSS is embedded inline so each file is self-contained for Playwright.
    """
    css_path = project_root / "static" / "style.css"
    if not css_path.exists():
        raise FileNotFoundError(f"CSS not found: {css_path}")
    css = css_path.read_text(encoding="utf-8")

    env = Environment(
        loader=FileSystemLoader(str(project_root / "templates")),
        autoescape=False,
    )
    template = env.get_template("instagram_post.html")

    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    # Build all display variants from the crossed stimulus factors
    factor_names  = list(STIMULUS_FACTORS.keys())
    factor_values = list(STIMULUS_FACTORS.values())
    display_variants = list(itertools.product(*factor_values)) if factor_values else [()]

    html_paths: list[Path] = []
    for row in rows:
        for variant in display_variants:
            variant_map = dict(zip(factor_names, variant))

            # Resolve anonymity
            anonymity = variant_map.get("comment_anonymity", "Not-anonymous")
            anonymous = anonymity.lower().startswith("anon")

            # Resolve engagement counts
            engagement = variant_map.get("post_engagement", "high").lower()
            likes, comments_count = ENGAGEMENT_COUNTS.get(engagement, ENGAGEMENT_COUNTS["high"])

            avatar_colour, commenter_username, post_age = _row_decoration(row, base_seed, anonymous)
            context = build_context(
                row, css, avatar_colour, commenter_username, post_age, likes, comments_count
            )

            slug  = "_".join(v.lower().replace(" ", "_").replace("-", "_") for v in variant)
            fname = f"{row.stimulus_id}_{slug}.html" if slug else f"{row.stimulus_id}.html"
            html_path = html_dir / fname
            html_path.write_text(template.render(**context), encoding="utf-8")
            html_paths.append(html_path)

    log.info("Rendered %d HTML stimuli → %s", len(html_paths), html_dir)
    return html_paths


# ── Playwright screenshot renderer ────────────────────────────────────────────

async def take_screenshots(
    html_paths: list[Path],
    png_dir: Path,
    batch_size: int = SCREENSHOT_BATCH_SIZE,
) -> None:
    """
    Render HTML files to HiDPI PNG using Playwright Chromium.

    Settings: 375×812 viewport, device_scale_factor=2, crops .ig-wrapper.
    Idempotent: skips files already rendered.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        log.error(
            "Playwright is not installed. "
            "Run: pip install playwright && playwright install chromium"
        )
        return

    png_dir.mkdir(parents=True, exist_ok=True)
    total = len(html_paths)
    done  = 0

    log.info("Rendering %d screenshots (batch=%d) …", total, batch_size)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()

        async def _render(html_path: Path) -> None:
            png_path = png_dir / (html_path.stem + ".png")
            if png_path.exists():
                return
            page = await browser.new_page(
                viewport=SCREENSHOT_VIEWPORT,
                device_scale_factor=2,
            )
            try:
                await page.goto(
                    f"file://{html_path.resolve()}",
                    wait_until="networkidle",
                )
                element = await page.query_selector(".ig-wrapper")
                if element:
                    await element.screenshot(path=str(png_path))
                else:
                    await page.screenshot(path=str(png_path), full_page=False)
            finally:
                await page.close()

        for batch_start in range(0, total, batch_size):
            batch = html_paths[batch_start : batch_start + batch_size]
            await asyncio.gather(*[_render(p) for p in batch])
            done += len(batch)
            log.info("  %d/%d screenshots rendered", done, total)

        await browser.close()

    log.info("PNGs saved → %s", png_dir)
