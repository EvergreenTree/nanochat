from __future__ import annotations

import math
import subprocess
from datetime import datetime
from pathlib import Path

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "slides" / "hackathon_pitch.pdf"
LOGO = ROOT / "dev" / "nanochat.png"
SCALING = ROOT / "dev" / "scaling_laws_jan26.png"

WIDTH = 13.333 * 72
HEIGHT = 7.5 * 72
MARGIN_X = 42
MARGIN_Y = 28

BG = HexColor("#F5F0E8")
INK = HexColor("#12202E")
MUTED = HexColor("#5C6673")
ACCENT = HexColor("#E45F3B")
ACCENT_2 = HexColor("#1B8A6B")
ACCENT_3 = HexColor("#F0C067")
PANEL = HexColor("#FFF9F1")
PANEL_DARK = HexColor("#162635")
WHITE = HexColor("#FFFFFF")


def git_value(args: list[str], fallback: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    except Exception:
        return fallback


BRANCH = git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"], "unknown-branch")
COMMIT = git_value(["git", "rev-parse", "--short", "HEAD"], "unknown")
BUILD_TIME = datetime.now().strftime("%Y-%m-%d %H:%M")


def box_shadow(c: canvas.Canvas, x: float, y: float, w: float, h: float, radius: float = 18) -> None:
    c.setFillColor(Color(0, 0, 0, alpha=0.08))
    c.roundRect(x + 6, y - 6, w, h, radius, stroke=0, fill=1)


def panel(c: canvas.Canvas, x: float, y: float, w: float, h: float, fill_color=WHITE, radius: float = 18) -> None:
    box_shadow(c, x, y, w, h, radius=radius)
    c.setFillColor(fill_color)
    c.roundRect(x, y, w, h, radius, stroke=0, fill=1)


def draw_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    *,
    font: str = "Helvetica",
    size: float = 14,
    color=INK,
) -> None:
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, text)


def draw_wrapped(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str = "Helvetica",
    size: float = 15,
    leading: float = 19,
    color=INK,
) -> float:
    lines = simpleSplit(text, font, size, width)
    text_obj = c.beginText(x, y)
    text_obj.setFont(font, size)
    text_obj.setLeading(leading)
    text_obj.setFillColor(color)
    for line in lines:
        text_obj.textLine(line)
    c.drawText(text_obj)
    return y - leading * len(lines)


def draw_bullets(
    c: canvas.Canvas,
    items: list[str],
    x: float,
    y: float,
    width: float,
    *,
    size: float = 15,
    leading: float = 20,
    bullet_color=ACCENT,
    text_color=INK,
) -> float:
    cursor = y
    for item in items:
        c.setFillColor(bullet_color)
        c.circle(x + 5, cursor + 4, 3, stroke=0, fill=1)
        cursor = draw_wrapped(
            c,
            item,
            x + 16,
            cursor,
            width - 16,
            font="Helvetica",
            size=size,
            leading=leading,
            color=text_color,
        )
        cursor -= 7
    return cursor


def header(c: canvas.Canvas, index: int, title: str, kicker: str) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, WIDTH, HEIGHT, stroke=0, fill=1)
    draw_text(c, kicker.upper(), MARGIN_X, HEIGHT - 34, font="Helvetica-Bold", size=10, color=ACCENT)
    draw_text(c, title, MARGIN_X, HEIGHT - 66, font="Helvetica-Bold", size=28, color=INK)
    c.setStrokeColor(HexColor("#D7CEC1"))
    c.setLineWidth(1)
    c.line(MARGIN_X, HEIGHT - 78, WIDTH - MARGIN_X, HEIGHT - 78)
    draw_text(c, f"{index:02d}", WIDTH - MARGIN_X - 20, HEIGHT - 34, font="Helvetica-Bold", size=11, color=MUTED)


def footer(c: canvas.Canvas, text: str = "EvergreenTree/nanochat | personal fork of karpathy/nanochat") -> None:
    draw_text(c, text, MARGIN_X, 15, size=9, color=MUTED)
    draw_text(c, f"{BRANCH} @ {COMMIT} | built {BUILD_TIME}", WIDTH - 262, 15, size=9, color=MUTED)


def stat_chip(c: canvas.Canvas, x: float, y: float, w: float, label: str, value: str, accent=ACCENT) -> None:
    c.setFillColor(accent)
    c.roundRect(x, y, w, 58, 18, stroke=0, fill=1)
    draw_text(c, label.upper(), x + 16, y + 39, font="Helvetica-Bold", size=10, color=WHITE)
    draw_text(c, value, x + 16, y + 16, font="Helvetica-Bold", size=18, color=WHITE)


def code_block(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, lines: list[str]) -> None:
    panel(c, x, y, w, h, fill_color=PANEL_DARK, radius=20)
    draw_text(c, title, x + 18, y + h - 24, font="Helvetica-Bold", size=12, color=ACCENT_3)
    text = c.beginText(x + 18, y + h - 48)
    text.setFont("Courier", 11)
    text.setLeading(15)
    text.setFillColor(WHITE)
    for line in lines:
        text.textLine(line)
    c.drawText(text)


def image_cover(c: canvas.Canvas, image_path: Path, x: float, y: float, w: float, h: float) -> None:
    if not image_path.exists():
        return
    image = ImageReader(str(image_path))
    iw, ih = image.getSize()
    scale = min(w / iw, h / ih)
    draw_w = iw * scale
    draw_h = ih * scale
    dx = x + (w - draw_w) / 2
    dy = y + (h - draw_h) / 2
    c.drawImage(image, dx, dy, draw_w, draw_h, mask="auto")


def slide_cover(c: canvas.Canvas) -> None:
    header(c, 1, "nanochat Precision Racecar", "GPU MODE x PyTorch Track 1 pitch")
    panel(c, 42, 78, 876, 376, fill_color=PANEL, radius=24)
    image_cover(c, LOGO, 58, 372, 280, 62)
    draw_wrapped(
        c,
        "A B300-ready product for controlled BF16 vs FP8 vs FP4 pretraining.",
        58,
        322,
        470,
        font="Helvetica-Bold",
        size=24,
        leading=28,
    )
    draw_wrapped(
        c,
        "Prestage assets, fail fast on Blackwell, train multi-node, and emit comparison artifacts a judge can read in one minute.",
        58,
        192,
        492,
        size=16,
        leading=22,
        color=MUTED,
    )
    stat_chip(c, 58, 84, 150, "flow", "prestage")
    stat_chip(c, 220, 84, 150, "flow", "smoke", accent=ACCENT_2)
    stat_chip(c, 382, 84, 150, "flow", "compare", accent=ACCENT_3)
    code_block(
        c,
        578,
        112,
        300,
        296,
        "Judge-facing output",
        [
            "BASE_DIR=/shared/nanochat/b300 \\",
            "RUN_NAME=track1-demo \\",
            "bash runs/fog_compare_b300.sh",
            "",
            "Artifacts:",
            "  comparisons/<run>/comparison.json",
            "  comparisons/<run>/comparison.md",
            "  base_checkpoints/<model_tag>/",
        ],
    )
    footer(c)
    c.showPage()


def slide_judges(c: canvas.Canvas) -> None:
    header(c, 2, "What Judges Actually Score", "Pitch framing")
    panel(c, 42, 96, 876, 362, fill_color=WHITE, radius=24)
    draw_text(c, "Strong hackathon demos look like products, not terminal archaeology.", 62, 414, font="Helvetica-Bold", size=24)
    draw_bullets(
        c,
        [
            "A clean runbook another team can execute without your verbal explanation.",
            "A smoke test that fails on the real reasons teams lose time: wrong GPU class, missing FA4, bad rendezvous, unsupported precision path.",
            "One artifact that makes the result legible: same tokenizer, same shard order, same seed, same batch schedule, only precision changes.",
            "A crisp claim: better throughput under controlled settings, with quality drift visible instead of hand-waved.",
        ],
        68,
        372,
        420,
        size=16,
        leading=22,
    )
    panel(c, 528, 124, 354, 300, fill_color=PANEL, radius=22)
    draw_text(c, "The one-line thesis", 550, 392, font="Helvetica-Bold", size=14, color=ACCENT)
    draw_wrapped(
        c,
        "Track 1 becomes a controlled product: same pipeline, same data, same seed, only precision changes.",
        550,
        344,
        290,
        font="Helvetica-Bold",
        size=19,
        leading=23,
    )
    draw_text(c, "Judge shortcuts we optimize for", 550, 222, font="Helvetica-Bold", size=14, color=ACCENT_2)
    draw_bullets(
        c,
        [
            "Can it start cleanly?",
            "Can I see the comparison in one file?",
        ],
        548,
        190,
        286,
        size=15,
        leading=20,
        bullet_color=ACCENT_2,
    )
    footer(c)
    c.showPage()


def slide_surface(c: canvas.Canvas) -> None:
    header(c, 3, "Product Surface", "What ships in the repo")
    card_w = 268
    gap = 18
    x0 = 42
    y = 120
    cards = [
        (
            "1. Prestage shared assets",
            ACCENT,
            [
                "runs/b300_prestage.sh",
                "",
                "Checks Blackwell runtime, pulls dataset shards, trains tokenizer, and leaves a clean BASE_DIR for the timed window.",
            ],
        ),
        (
            "2. Fail fast before scoring",
            ACCENT_2,
            [
                "runs/b300_smoke.sh",
                "",
                "Exercises the intended FP8 stack on one node so the real cluster slot is not spent discovering missing FA4 or FP8 incompatibilities.",
            ],
        ),
        (
            "3. Train and compare",
            ACCENT_3,
            [
                "runs/b300_train.sh",
                "runs/b300_eval.sh",
                "runs/fog_compare_b300.sh",
                "",
                "Produces checkpoints plus comparison.json and comparison.md for judge-ready inspection.",
            ],
        ),
    ]
    for idx, (title, accent, lines) in enumerate(cards):
        x = x0 + idx * (card_w + gap)
        panel(c, x, y, card_w, 310, fill_color=WHITE, radius=22)
        c.setFillColor(accent)
        c.roundRect(x, y + 268, card_w, 42, 18, stroke=0, fill=1)
        draw_text(c, title, x + 16, y + 283, font="Helvetica-Bold", size=16, color=WHITE)
        text = c.beginText(x + 16, y + 236)
        text.setFont("Courier", 11 if idx != 2 else 10)
        text.setLeading(14)
        text.setFillColor(INK)
        for line in lines[: max(2, len(lines) - 1)]:
            text.textLine(line)
        c.drawText(text)
        draw_wrapped(c, lines[-1], x + 16, y + 112, card_w - 32, size=14, leading=18, color=MUTED)
    footer(c)
    c.showPage()


def slide_control(c: canvas.Canvas) -> None:
    header(c, 4, "Controlled Precision Lab", "Why the comparison is believable")
    panel(c, 42, 100, 438, 350, fill_color=WHITE, radius=24)
    draw_text(c, "Fixed across all three arms", 62, 414, font="Helvetica-Bold", size=22)
    draw_bullets(
        c,
        [
            "Tokenizer and dataset shards",
            "Random seed and shard order",
            "Architecture family and depth",
            "Total batch size and token budget",
            "Optimizer schedule and eval cadence",
            "Comparison metrics: tok/sec, step time, wall time, val_bpb",
        ],
        66,
        376,
        350,
        size=16,
        leading=22,
    )
    panel(c, 510, 100, 408, 350, fill_color=PANEL, radius=24)
    draw_text(c, "Only one variable moves", 532, 414, font="Helvetica-Bold", size=22)
    stat_chip(c, 532, 320, 118, "recipe", "bf16", accent=INK)
    stat_chip(c, 664, 320, 118, "recipe", "fp8", accent=ACCENT)
    stat_chip(c, 796, 320, 118, "recipe", "fp4", accent=ACCENT_2)
    draw_wrapped(
        c,
        "The deck claim is simple: when a judge asks whether FP8 or FP4 actually helped, we can answer with matched runs rather than changing five knobs at once.",
        532,
        274,
        338,
        size=16,
        leading=22,
        color=MUTED,
    )
    code_block(
        c,
        532,
        134,
        338,
        112,
        "Artifact schema",
        [
            "{",
            '  "recipe": "fp8_full",',
            '  "tok_per_sec": 0.0,',
            '  "step_time_s": 0.0,',
            '  "val_bpb": 0.0',
            "}",
        ],
    )
    footer(c)
    c.showPage()


def slide_credibility(c: canvas.Canvas) -> None:
    header(c, 5, "Why This Is Credible", "Built on real repo surface")
    panel(c, 42, 110, 398, 338, fill_color=WHITE, radius=24)
    draw_text(c, "This is not a slideware rewrite.", 62, 410, font="Helvetica-Bold", size=24)
    draw_bullets(
        c,
        [
            "The fork already carries B300 prestage, smoke, train, and eval entrypoints.",
            "The FOG family exists separately from default nanochat, so the precision experiment does not get polluted by unrelated architecture tricks.",
            "The comparison harness already writes machine-readable artifacts for side-by-side throughput and quality.",
            "The upstream nanochat story remains intact, which makes the fork legible to other contributors.",
        ],
        66,
        370,
        330,
        size=15,
        leading=20,
    )
    panel(c, 468, 110, 450, 338, fill_color=PANEL, radius=24)
    draw_text(c, "Upstream credibility + Blackwell extension", 490, 410, font="Helvetica-Bold", size=22)
    image_cover(c, SCALING, 490, 246, 406, 118)
    draw_wrapped(
        c,
        "nanochat already has a speedrun identity. The product pitch is that this fork extends that identity to Blackwell-era precision experiments without turning the repo into an opaque research pile.",
        490,
        214,
        386,
        size=15,
        leading=21,
        color=MUTED,
    )
    stat_chip(c, 490, 136, 124, "base", "nanochat", accent=INK)
    stat_chip(c, 628, 136, 124, "fork", "B300", accent=ACCENT)
    stat_chip(c, 766, 136, 124, "story", "product", accent=ACCENT_2)
    footer(c)
    c.showPage()


def slide_demo(c: canvas.Canvas) -> None:
    header(c, 6, "Live Demo In 60 Seconds", "What we would show the judges")
    code_block(
        c,
        42,
        138,
        410,
        280,
        "1. Bring the node up cleanly",
        [
            "BASE_DIR=/shared/nanochat/b300 \\",
            "bash runs/b300_smoke.sh",
            "",
            "# proves:",
            "# - Blackwell detected",
            "# - FlashAttention-4 active",
            "# - precision stack usable",
        ],
    )
    code_block(
        c,
        478,
        138,
        440,
        280,
        "2. Show the product artifact",
        [
            "cat $BASE_DIR/comparisons/track1/comparison.md",
            "",
            "| recipe | tok/sec | step_s | val_bpb |",
            "| bf16   | ...     | ...    | ...     |",
            "| fp8    | ...     | ...    | ...     |",
            "| fp4    | ...     | ...    | ...     |",
        ],
    )
    panel(c, 42, 86, 876, 36, fill_color=ACCENT, radius=18)
    draw_text(
        c,
        "The visual proof is compact: one command to validate the system, one file to understand the result.",
        62,
        98,
        font="Helvetica-Bold",
        size=16,
        color=WHITE,
    )
    footer(c)
    c.showPage()


def slide_close(c: canvas.Canvas) -> None:
    header(c, 7, "Why This Can Win", "Closing ask")
    panel(c, 42, 90, 876, 370, fill_color=PANEL_DARK, radius=28)
    draw_text(c, "Most teams will present a run.", 66, 396, font="Helvetica-Bold", size=30, color=WHITE)
    draw_text(c, "We can present a reusable training product.", 66, 360, font="Helvetica-Bold", size=30, color=ACCENT_3)
    draw_bullets(
        c,
        [
            "Judges can see the operational path from prestage to comparison without reverse-engineering our terminals.",
            "Sponsors can imagine this becoming a public Blackwell benchmark, not just a hackathon artifact.",
            "The prize hardware extends the same story cleanly: larger matched sweeps, stronger evidence, public reproducibility.",
        ],
        70,
        314,
        520,
        size=17,
        leading=23,
        bullet_color=ACCENT_3,
        text_color=WHITE,
    )
    panel(c, 632, 120, 240, 268, fill_color=WHITE, radius=22)
    draw_text(c, "Repo", 654, 350, font="Helvetica-Bold", size=12, color=ACCENT)
    draw_wrapped(
        c,
        "EvergreenTree/nanochat",
        654,
        322,
        176,
        font="Helvetica-Bold",
        size=18,
        leading=22,
        color=INK,
    )
    draw_text(c, "Pitch line", 654, 268, font="Helvetica-Bold", size=12, color=ACCENT_2)
    draw_wrapped(
        c,
        "A Blackwell-ready control tower for fair low-precision pretraining comparisons.",
        654,
        242,
        182,
        size=15,
        leading=20,
        color=MUTED,
    )
    draw_text(c, "Ask", 654, 184, font="Helvetica-Bold", size=12, color=ACCENT)
    draw_wrapped(
        c,
        "Back the team shipping product, not just a run.",
        654,
        158,
        182,
        size=14,
        leading=18,
        color=MUTED,
    )
    footer(c)
    c.showPage()


def build(output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT))
    c.setTitle("nanochat Track 1 Hackathon Pitch")
    c.setAuthor("Changqing Fu / EvergreenTree")
    c.setSubject("B300 Track 1 product pitch for nanochat precision comparisons")
    c.setCreator("slides/hackathon_pitch.py")
    slide_cover(c)
    slide_judges(c)
    slide_surface(c)
    slide_control(c)
    slide_credibility(c)
    slide_demo(c)
    slide_close(c)
    c.save()


if __name__ == "__main__":
    build()
    print(f"Wrote {OUTPUT}")
