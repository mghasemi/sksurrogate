#!/usr/bin/env python3
"""Generate assets/og-card.png — the 1200x630 social/LLM preview card.

Copy rule (HANDOFF-website-seo.md, stop-and-ask #5): the headline is the page's
existing <h1>, verbatim. Line breaks are presentation only; no word is added,
removed or reordered. The bottom line is the page's own meta strip.

Requires: Pillow, and the Inter / Fira Mono OTFs present on this machine
(/usr/share/fonts/opentype/{inter,fira}). Run from the repo root:

    python scripts/make_og_card.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = (11, 15, 20)
INK = (231, 236, 244)
DIM = (231, 236, 244, 158)
FAINT = (231, 236, 244, 102)
ULTRA = (138, 165, 255)
EMBER = (224, 122, 60)
TEAL = (79, 179, 191)
LINE = (235, 238, 245, 33)

FONT_DIR_I = "/usr/share/fonts/opentype/inter"
FONT_DIR_F = "/usr/share/fonts/opentype/fira"

# h1 text, verbatim from index.html (U+2011 non-breaking hyphen preserved)
HEAD = [
    ("From raw dataset to deployed,", INK),
    ("drift\u2011monitored model \u2014", INK),
    ("in one scikit\u2011learn\u2011native environment.", ULTRA),
]


def tracked(draw, xy, text, font, fill, tracking: float = 0.0):
    """Draw text with letter-spacing (PIL has no native tracking)."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking


def mark(draw, x, y, scale=0.52):
    """The brand mark: six nested ellipses + ember dot, from assets/mark.svg
    (viewBox 0 0 120 132). Drawn as outlines to keep the file type-free."""
    widths = [6, 6, 6, 6, 6, 6]
    ellipses = [(60, 14, 40, 10.4), (60, 29.8, 35.4, 9.2), (60, 45.6, 31.2, 8.1),
                (60, 61.4, 27.3, 7.1), (60, 77.2, 23.6, 6.1), (60, 93, 20.3, 5.3)]
    colors = [ULTRA, (91, 124, 250), (79, 179, 191), TEAL, TEAL, (224, 122, 60)]
    for (cx, cy, rx, ry), w, c in zip(ellipses, widths, colors):
        draw.ellipse([x + (cx - rx) * scale, y + (cy - ry) * scale,
                      x + (cx + rx) * scale, y + (cy + ry) * scale],
                     outline=c + (255,), width=max(1, round(w * scale)))
    r = 9 * scale
    draw.ellipse([x + 60 * scale - r, y + 120 * scale - r,
                  x + 60 * scale + r, y + 120 * scale + r], fill=EMBER + (255,))


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")

    # hairline frame, echoing the site's 1px rules
    d.rectangle([24, 24, W - 25, H - 25], outline=LINE, width=1)

    f_head = ImageFont.truetype(os.path.join(FONT_DIR_I, "Inter-Light.otf"), 46)
    f_word = ImageFont.truetype(os.path.join(FONT_DIR_I, "Inter-SemiBold.otf"), 27)
    f_mono = ImageFont.truetype(os.path.join(FONT_DIR_F, "FiraMono-Regular.otf"), 14)
    f_meta = ImageFont.truetype(os.path.join(FONT_DIR_F, "FiraMono-Regular.otf"), 17)

    # brand block
    mark(d, 72, 62)
    d.text((132, 74), "SKSurrogate", font=f_word, fill=INK)
    tracked(d, (72, 168), "SCIENTIFIC MLOPS \u00b7 OPEN SOURCE", f_mono, FAINT, 2.6)

    # headline (verbatim h1)
    y = 222
    for text, color in HEAD:
        d.text((72, y), text, font=f_head, fill=color)
        y += 62

    # bottom rule + meta line
    d.line([72, H - 116, W - 72, H - 116], fill=LINE, width=1)
    d.text((72, H - 88), "MIT", font=f_meta, fill=INK)
    tracked(d, (72 + 46, H - 86), "\u00b7", f_meta, FAINT, 0)
    d.text((72 + 68, H - 88), "pip install sksurrogate", font=f_meta, fill=DIM)

    out = "assets/og-card.png"
    img.save(out, "PNG", optimize=True)
    print(out, img.size, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
