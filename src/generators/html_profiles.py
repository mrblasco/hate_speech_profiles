"""
Instagram-style profile page renderer.

Converts Profile objects into HTML profile pages using
templates/instagram_profile.html + static/profile_style.css,
then optionally fires Playwright for HiDPI PNG screenshots.

All numeric decoration (followers, following, post count) and visual choices
(avatar colour, grid gradients) are derived deterministically from the run seed.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.models import Profile
from src.utils.logging_utils import get_logger
from src.utils.seeds import derive_seed
from src.generators.html_stimuli import (
    AVATAR_COLOURS,
    TOPIC_GRADIENTS,
    TOPIC_EMOJIS,
    SCREENSHOT_BATCH_SIZE,
    SCREENSHOT_VIEWPORT,
    _FALLBACK_GRADIENT,
    _pick,
    _avatar_initials,
    take_screenshots,
)

log = get_logger("generators.html_profiles")

# ── Visual constants ──────────────────────────────────────────────────────────

FOLLOWER_COUNTS = [
    "847", "1.2K", "2.9K", "4.1K", "531", "1.8K",
    "3.2K", "723", "2.1K", "5.9K", "1.0K", "967",
]

FOLLOWING_COUNTS = [
    "234", "412", "189", "567", "301", "445",
    "278", "623", "198", "389", "441", "356",
]

POST_COUNTS = [
    "23", "41", "67", "18", "34", "52",
    "89", "29", "47", "71", "38", "44",
]

RELIGION_EMOJIS = {
    "Muslim":    "☪️",
    "Christian": "✝️",
    "Jewish":    "✡️",
}

# Pool of gradients used to fill the 3×3 post grid
GRID_GRADIENTS = [
    "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
    "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
    "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
    "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
    "linear-gradient(135deg, #30cfd0 0%, #330867 100%)",
    "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)",
    "linear-gradient(135deg, #fd7043 0%, #ff7043 100%)",
    "linear-gradient(135deg, #e96c5a 0%, #e91e63 100%)",
    "linear-gradient(135deg, #2196f3 0%, #21cbf3 100%)",
]

GRID_EMOJIS = [
    "🌍", "✊", "🕌", "🌱", "💉", "🏴",
    "🌿", "🔥", "⚡", "🎯", "💡", "🗣️",
    "📢", "🤝", "🌐", "🕊️", "⚖️", "🧬",
]

# Interest labels shown under story-highlight circles
_HIGHLIGHT_EMOJIS = ["📌", "💬", "📸", "🎵", "✈️", "📖", "🎨", "🏃"]


# ── Context builder ───────────────────────────────────────────────────────────

def _enum_str(v: object) -> str:
    return v.value if hasattr(v, "value") else str(v) if v is not None else ""


def build_profile_context(profile: object, topic: str, css: str, base_seed: int) -> dict:
    """
    Build the Jinja2 context dict for one profile page.

    `profile` may be a Profile or a StimulusRow — any object carrying the
    profile-level attributes (profile_id, username, display_name, bio, values,
    religion, country_of_origin). `interests` is optional; when absent the
    story-highlight row is simply left empty.
    """
    pid = profile.profile_id

    avatar_colour = _pick(AVATAR_COLOURS,   derive_seed(base_seed, "avatar",    pid))
    followers     = _pick(FOLLOWER_COUNTS,  derive_seed(base_seed, "followers", pid))
    following     = _pick(FOLLOWING_COUNTS, derive_seed(base_seed, "following", pid))
    post_count    = _pick(POST_COUNTS,      derive_seed(base_seed, "posts",     pid))

    interests = getattr(profile, "interests", None) or []
    highlight_items = []
    for i, interest in enumerate(interests[:4]):
        label = interest.split()[0].rstrip(",;")
        emoji    = _pick(GRID_EMOJIS,    derive_seed(base_seed, f"hi{i}",  pid))
        gradient = _pick(GRID_GRADIENTS, derive_seed(base_seed, f"hg{i}",  pid))
        highlight_items.append({"emoji": emoji, "label": label, "gradient": gradient})

    grid_items = [
        {
            "gradient": _pick(GRID_GRADIENTS, derive_seed(base_seed, f"grid{i}",    pid)),
            "emoji":    _pick(GRID_EMOJIS,    derive_seed(base_seed, f"gremoji{i}", pid)),
        }
        for i in range(9)
    ]

    religion_str = _enum_str(getattr(profile, "religion", None))
    country_str  = _enum_str(getattr(profile, "country_of_origin", None))
    values_str   = _enum_str(getattr(profile, "values", None))

    return {
        "css":              css,
        "lang":             "en",
        "username":         profile.username,
        "display_name":     profile.display_name,
        "avatar_initials":  _avatar_initials(profile.display_name),
        "avatar_colour":    avatar_colour,
        "bio":              profile.bio,
        "post_count":       post_count,
        "followers":        followers,
        "following":        following,
        "topic":            topic,
        "topic_emoji":      TOPIC_EMOJIS.get(topic, "📱"),
        "values":           values_str,
        "religion":         religion_str,
        "religion_emoji":   RELIGION_EMOJIS.get(religion_str, "🙏"),
        "country_of_origin": country_str,
        "highlight_items":  highlight_items,
        "grid_items":       grid_items,
    }


# ── HTML generation ───────────────────────────────────────────────────────────

def generate_profile_pages(
    profile_topic_pairs: list[tuple[object, str]],
    output_dir: Path,
    project_root: Path,
    base_seed: int,
) -> list[Path]:
    """
    Render one HTML profile page per profile.

    `profile_topic_pairs` is a list of ``(profile_like, topic_str)`` where
    profile_like is a Profile or StimulusRow (or any object with the profile
    attributes) and topic_str is the plain topic string (e.g. "climate").

    Filename pattern: {profile_id}_profile.html
    Output directory: output_dir / "html_profiles"
    CSS is embedded inline so each file is self-contained for Playwright.
    """
    css_path = project_root / "static" / "profile_style.css"
    if not css_path.exists():
        raise FileNotFoundError(f"Profile CSS not found: {css_path}")
    css = css_path.read_text(encoding="utf-8")

    env = Environment(
        loader=FileSystemLoader(str(project_root / "templates")),
        autoescape=False,
    )
    template = env.get_template("instagram_profile.html")

    html_dir = output_dir / "html_profiles"
    html_dir.mkdir(parents=True, exist_ok=True)

    html_paths: list[Path] = []
    for profile, topic in profile_topic_pairs:
        context   = build_profile_context(profile, topic, css, base_seed)
        html_path = html_dir / f"{profile.profile_id}_profile.html"
        html_path.write_text(template.render(**context), encoding="utf-8")
        html_paths.append(html_path)

    log.info("Rendered %d profile pages → %s", len(html_paths), html_dir)
    return html_paths
