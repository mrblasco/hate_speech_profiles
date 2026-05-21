"""
HTML (and optional PNG) stimulus renderer.

Converts final StimulusRow objects into Instagram-style HTML files using
templates/instagram_post.html + static/style.css, then optionally fires
Playwright to produce HiDPI PNG screenshots.

For each (stimulus, engagement_variant) pair two HTML files are produced:
  *_with_comment.html  — full stimulus including the comment
  *_no_comment.html    — post only, comment block hidden

Topic gradients, emojis, and target groups are read from configs/topics.yaml
via TopicRegistry — no hardcoded dicts in this file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.models import StimulusRow
from src.utils.io import TopicRegistry
from src.utils.logging_utils import get_logger
from src.utils.seeds import derive_seed

log = get_logger("generators.html")

# ── Visual constants ──────────────────────────────────────────────────────────

AVATAR_COLOURS = [
    "#E91E8C", "#2196F3", "#9C27B0", "#FF5722",
    "#4CAF50", "#FF9800", "#00BCD4", "#F44336",
    "#3F51B5", "#009688", "#8BC34A", "#FFC107",
]

COMMENTER_USERNAMES = [
    "user48291",       "real_talk_99",    "truth_seeker_x",  "anonymous_voice",
    "just_sayin_2024", "no_filter_guy",   "freedom_speaker", "real_deal_22",
    "speaking_facts_", "open_minded_99",  "debate_this_now", "commentator45",
    "plaintruth2024",  "honest_guy_12",   "voice_of_reason", "citizen_free",
    "noreservations_", "outsider_view",   "justmyopinion99", "the_watchman_x",
    "critical_thinker","questioning_all", "skeptic_online",  "frank_comments",
    "telling_it_real", "common_sense_01", "facts_not_feels", "reality_check_",
]

POST_AGES = [
    "2 HOURS AGO", "3 HOURS AGO", "4 HOURS AGO",
    "5 HOURS AGO", "6 HOURS AGO", "8 HOURS AGO",
]

# All 9 engagement rows. Tuple: (engagement_level, variant_index, likes, comments_count)
ENGAGEMENT_ROWS: list[tuple[str, int, int, int]] = [
    ("low",    1,  12,   2),
    ("low",    2,  8,    3),
    ("low",    3,  21,   4),
    ("medium", 1,  187,  18),
    ("medium", 2,  234,  24),
    ("medium", 3,  312,  31),
    ("high",   1,  2934, 156),
    ("high",   2,  3847, 203),
    ("high",   3,  5123, 189),
]

_FALLBACK_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
_FALLBACK_EMOJI    = "📱"

CAPTION_PREVIEW_LENGTH = 120
SCREENSHOT_BATCH_SIZE  = 20
SCREENSHOT_VIEWPORT    = {"width": 375, "height": 812}

_registry: TopicRegistry | None = None


def init_registry(registry: TopicRegistry) -> None:
    """Call once at pipeline startup to inject the shared TopicRegistry."""
    global _registry
    _registry = registry


def _topic_gradient(topic: str) -> str:
    if _registry:
        try:
            return _registry.gradient(topic)
        except KeyError:
            pass
    return _FALLBACK_GRADIENT


def _topic_emoji(topic: str) -> str:
    if _registry:
        try:
            return _registry.emoji(topic)
        except KeyError:
            pass
    return _FALLBACK_EMOJI


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick(pool: list, seed_int: int):
    return pool[seed_int % len(pool)]


def _avatar_initials(display_name: str) -> str:
    return "".join(w[0].upper() for w in display_name.split()[:2])


# ── Context builder ───────────────────────────────────────────────────────────

def _row_decoration(row: StimulusRow, base_seed: int) -> tuple[str, str, str]:
    """Return (avatar_colour, commenter_username, post_age) — stable across variants."""
    avatar_colour      = _pick(AVATAR_COLOURS,      derive_seed(base_seed, "avatar",    row.profile_id))
    commenter_username = _pick(COMMENTER_USERNAMES, derive_seed(base_seed, "commenter", row.comment_id))
    post_age           = _pick(POST_AGES,            derive_seed(base_seed, "age",       row.stimulus_id))
    return avatar_colour, commenter_username, post_age


def build_context(
    row: StimulusRow,
    css: str,
    avatar_colour: str,
    commenter_username: str,
    post_age: str,
    likes: int,
    comments_count: int,
    show_comment: bool,
    anonymous_display_name: str = "anonymous_user",
) -> dict:
    topic = row.topic.value if hasattr(row.topic, "value") else str(row.topic)

    n = f"{comments_count:,}"
    likes_fmt = f"{likes:,}"

    caption = row.caption
    caption_truncated = len(caption) > CAPTION_PREVIEW_LENGTH
    caption_preview = caption[:CAPTION_PREVIEW_LENGTH].rstrip() + "…" if caption_truncated else caption

    # Respect anonymity: hide real username when anonymous
    anonymity = (row.anonymity or "named").lower()
    display_commenter = (
        anonymous_display_name if anonymity == "anonymous" else commenter_username
    )

    return {
        "css":                css,
        "lang":               "en",
        "username":           row.username,
        "display_name":       row.display_name,
        "avatar_initials":    _avatar_initials(row.display_name),
        "avatar_colour":      avatar_colour,
        "target_message":     row.caption,
        "topic_gradient":     _topic_gradient(topic),
        "topic_emoji":        _topic_emoji(topic),
        "commenter_username": display_commenter,
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
        "show_comment":       show_comment,
    }


# ── HTML generation ───────────────────────────────────────────────────────────

def generate_html_stimuli(
    rows: list[StimulusRow],
    output_dir: Path,
    project_root: Path,
    base_seed: int,
    likes_filter: str | None = None,
    anonymous_display_name: str = "anonymous_user",
) -> list[Path]:
    """
    Render HTML stimuli for each StimulusRow.

    For every (row, engagement_variant) pair, two HTML files are written:
      {stimulus_id}_{level}{variant}_with_comment.html
      {stimulus_id}_{level}{variant}_no_comment.html

    Parameters
    ----------
    rows:                   Passed StimulusRow objects.
    output_dir:             Parent of the html/ subdirectory.
    project_root:           Repo root; templates/ and static/ resolved from here.
    base_seed:              Master run seed for deterministic decoration.
    likes_filter:           When set ("low"/"medium"/"high"), only render the 3
                            engagement variants at that level (not all 9).
    anonymous_display_name: Username shown when anonymity == "anonymous".

    Returns
    -------
    All written HTML file paths.
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

    # Filter engagement rows by likes level when requested
    engagement_rows = ENGAGEMENT_ROWS
    if likes_filter:
        level_map = {"low": "low", "mid": "medium", "medium": "medium", "high": "high"}
        target_level = level_map.get(likes_filter.lower(), likes_filter.lower())
        engagement_rows = [r for r in ENGAGEMENT_ROWS if r[0] == target_level]
        if not engagement_rows:
            log.warning("likes_filter=%r matched no engagement rows; using all", likes_filter)
            engagement_rows = ENGAGEMENT_ROWS

    html_paths: list[Path] = []
    for row in rows:
        avatar_colour, commenter_username, post_age = _row_decoration(row, base_seed)
        for level, variant_idx, likes, comments_count in engagement_rows:
            base_ctx = dict(
                row=row, css=css,
                avatar_colour=avatar_colour, commenter_username=commenter_username,
                post_age=post_age, likes=likes, comments_count=comments_count,
                anonymous_display_name=anonymous_display_name,
            )
            for show_comment, suffix in ((True, "with_comment"), (False, "no_comment")):
                ctx = build_context(show_comment=show_comment, **base_ctx)
                fname = f"{row.stimulus_id}_{level}{variant_idx}_{suffix}.html"
                html_path = html_dir / fname
                html_path.write_text(template.render(**ctx), encoding="utf-8")
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
