"""Regenerates the dashboard icons in nod/ui/. Needs Pillow (dev-only: `pip install pillow`; not needed to run Nod)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "nod" / "ui"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
WHITE, ORANGE, BLACK = (224, 224, 224), (245, 74, 0), (0, 0, 0)


def icon(size, rounded=True):
    s = size * 4                                        # draw big, shrink for smooth edges
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if rounded:
        d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.225), fill=BLACK)
    else:
        d.rectangle((0, 0, s, s), fill=BLACK)           # maskable icons are cropped by the OS, so fill edge to edge
    font = ImageFont.truetype(FONT, int(s * 0.60), index=1)   # index 1 = Bold in this .ttc
    box = d.textbbox((0, 0), "N", font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x, y = (s - w) / 2 - box[0] - s * 0.03, (s - h) / 2 - box[1] + s * 0.03
    d.text((x, y), "N", font=font, fill=WHITE)
    r = s * 0.065                                       # the orange "nod" dot, riding the top-right of the N
    cx, cy = x + box[0] + w + s * 0.02, y + box[1] - s * 0.005
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ORANGE)
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    icon(512).save(OUT / "icon-512.png")
    icon(192).save(OUT / "icon-192.png")
    icon(512, rounded=False).save(OUT / "icon-512-maskable.png")
    print("wrote icons to", OUT)
