#!/usr/bin/env python3
"""Write scripts/banner.sh, the start.sh banner: Mimir (the wise head Odin consults) above Mimisbrunnr, the well of
wisdom, with Odin's pledged eye glowing in its water and a root of Yggdrasil beneath, beside ALEXANDER YW in a chunky
pixel font with a drop shadow, on an aurora gradient. Everything is drawn here, two pixels a character cell (the upper
half block with a foreground and a background colour), so the art needs a truecolour terminal.

Usage: python3 -I scripts/make_banner.py > scripts/banner.sh
"""

SUBTITLE = "Mimir 1.8 · one DGX Spark"

# Mimir's head, face on, as the left half of each row (mirrored): h hair and beard, w the beard's sheen, s skin,
# b brows, e the eyes, n the nose, m the mouth.
HEAD = """
............hhhhh
..........hhhhhhh
.........hhhhhhhh
........hhhhhhhhh
........hhsssssss
.......hhssssssss
.......hhsbbbbsss
.......hssseessss
.......hssssssssn
.......hhsssssssn
.......hhssshhhhh
........hhhhhhmmm
........hhhwhhhhh
.........hhwhhwhh
.........hwhhwhhw
..........hwhhwhh
..........whhwhhw
...........hwhhwh
...........whhwhh
............hwhhw
............whhwh
.............hwhh
..............whh
...............hw
................h
"""

# The well below it, whole rows (Odin's one eye breaks the symmetry): k its stones, S their joints, q the water,
# o the eye, t the root.
WELL = """
....kkkkkkkkkkkkkkkkkkkkkkkkkk....
...kqqqqqqqqqqqqqqqqqqqqqqqqqqk...
...kqqqqqqqqqqqqqqqqqqqqooqqqqk...
...kkqqqqqqqqqqqqqqqqqqqqqqqqkk...
....kkkkkkkkkkkkkkkkkkkkkkkkkk....
....kSkkkSkkkSkkkSkkkSkkkSkkkk....
...tttttttttttttttttttttttttttt...
"""

# A 7 x 9 pixel font, two-pixel strokes.
FONT = {
    "A": ["..###..", ".##.##.", "##...##", "##...##", "#######", "#######", "##...##", "##...##", "##...##"],
    "L": ["##.....", "##.....", "##.....", "##.....", "##.....", "##.....", "##.....", "#######", "#######"],
    "E": ["#######", "#######", "##.....", "##.....", "######.", "##.....", "##.....", "#######", "#######"],
    "X": ["##...##", "##...##", ".##.##.", ".##.##.", "..###..", ".##.##.", ".##.##.", "##...##", "##...##"],
    "N": ["##...##", "###..##", "###..##", "####.##", "##.####", "##..###", "##..###", "##...##", "##...##"],
    "D": ["#####..", "######.", "##..###", "##...##", "##...##", "##...##", "##..###", "######.", "#####.."],
    "R": ["######.", "#######", "##...##", "##...##", "######.", "#####..", "##.###.", "##..###", "##...##"],
    "Y": ["##...##", "##...##", ".##.##.", ".##.##.", "..###..", "..###..", "..###..", "..###..", "..###.."],
    "W": ["##...##", "##...##", "##...##", "##.#.##", "##.#.##", "#######", "###.###", "##...##", "#.....#"],
}

AURORA = [(0.0, (90, 255, 170)), (0.35, (40, 214, 232)), (0.7, (86, 140, 255)), (1.0, (178, 102, 255))]
SHADOW = (28, 36, 74)
MIMIR_COLOURS = {
    "s": (206, 186, 168),   # weathered skin
    "b": (150, 154, 178),   # silver brows
    "e": (255, 198, 64),    # gold eyes, as Munin's
    "n": (172, 150, 136),   # the nose's shade
    "m": (92, 72, 84),
    "S": (70, 72, 94),      # the stones' joints
    "o": (255, 198, 64),    # Odin's eye in the water
    "t": (104, 82, 64),     # Yggdrasil's root, Munin's branch
}
ESC = "\x1b"


def mix(stops, t):
    t = min(1.0, max(0.0, t))
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
    return stops[-1][1]


def mimir():
    """Mimir as a grid of colours (None: empty): silver hair and beard darkening downward with an aurora sheen,
    stone that darkens toward its foot, and the well's water an aurora band."""
    rows = [half + half[::-1] for half in HEAD.strip("\n").split("\n")] + WELL.strip("\n").split("\n")
    h, w = len(rows), max(map(len, rows))
    grid = []
    for y, row in enumerate(rows):
        out = []
        for x, ch in enumerate(row.ljust(w, ".")):
            t = y / (h - 1)
            if ch == "h":
                out.append(mix([(0.0, (178, 184, 206)), (1.0, (96, 98, 140))], t))
            elif ch == "w":
                out.append(mix([(0.0, (58, 150, 176)), (1.0, (118, 84, 196))], (y - 10) / 14))
            elif ch == "k":
                out.append(mix([(0.0, (112, 116, 138)), (1.0, (84, 86, 110))], (y - 25) / 5))
            elif ch == "q":
                out.append(mix(AURORA, x / (w - 1)))
            elif ch in MIMIR_COLOURS:
                out.append(MIMIR_COLOURS[ch])
            else:
                out.append(None)
        grid.append(out)
    return grid


def name(lines):
    """Rows of letters as one grid: aurora along the diagonal of the whole block, a navy shadow one pixel down-right."""
    words = []
    for text in lines:
        cols = []
        for i, ch in enumerate(text):
            if i:
                cols.append(["."] * 9)
            glyph = FONT[ch]
            for x in range(7):
                cols.append([glyph[y][x] for y in range(9)])
        words.append(["".join(c[y] for c in cols) for y in range(9)])
    width = max(len(w[0]) for w in words) + 1
    pix = []
    for i, w in enumerate(words):
        if i:
            pix += ["." * width] * 2
        pix += [r.ljust(width, ".") for r in w] + ["." * width]
    h = len(pix)
    grid = []
    for y in range(h):
        out = []
        for x in range(width):
            if pix[y][x] == "#":
                out.append(mix(AURORA, (x + 0.6 * y) / (width + 0.6 * h)))
            elif x and y and pix[y - 1][x - 1] == "#":
                out.append(SHADOW)
            else:
                out.append(None)
        grid.append(out)
    return grid


def cells(grid):
    """Pixel rows two at a time as character cells: (char, foreground, background)."""
    if len(grid) % 2:
        grid = grid + [[None] * len(grid[0])]
    lines = []
    for y in range(0, len(grid), 2):
        row = []
        for top, bottom in zip(grid[y], grid[y + 1]):
            if top is None and bottom is None:
                row.append((" ", None, None))
            elif bottom is None:
                row.append(("▀", top, None))
            elif top is None:
                row.append(("▄", bottom, None))
            elif top == bottom:
                row.append(("█", top, None))
            else:
                row.append(("▀", top, bottom))
        lines.append(row)
    return lines


def render(row):
    s, cur = "", (None, None)
    for ch, fg, bg in row:
        if (fg, bg) != cur:
            s += ESC + "[0m"
            if bg == "bold":    # text: bold, no background
                s += ESC + "[1m"
            if fg:
                s += ESC + "[38;2;%d;%d;%dm" % fg
            if bg and bg != "bold":
                s += ESC + "[48;2;%d;%d;%dm" % bg
            cur = (fg, bg)
        s += ch
    return (s + ESC + "[0m").rstrip() if cur != (None, None) else s.rstrip()


def credit(width):
    """\"TensorFold by Ash Hart\" centred in `width`: the engine and its author in bold aurora, "by" dimmed."""
    engine, author = "TensorFold", "Ash Hart"
    text = [(ch, mix(AURORA, 0.5 * i / (len(engine) - 1)), "bold") for i, ch in enumerate(engine)]
    text += [(ch, (150, 150, 170) if ch != " " else None, None) for ch in " by "]
    text += [(ch, mix(AURORA, 0.5 + 0.5 * i / (len(author) - 1)) if ch != " " else None, "bold" if ch != " " else None)
             for i, ch in enumerate(author)]
    pad = max(0, (width - len(text)) // 2)
    return [(" ", None, None)] * pad + text + [(" ", None, None)] * max(0, width - pad - len(text))


def compose(words, gap=4):
    left = cells(mimir())
    lw = len(left[0])
    left += [[(" ", None, None)] * lw, credit(lw)]
    right = cells(name(words))
    rw = len(right[0])
    sub = SUBTITLE.rjust(rw)
    right += [[(" ", None, None)] * rw, [(ch, (150, 150, 170) if ch != " " else None, None) for ch in sub]]
    top = 1
    n = max(len(left), top + len(right))
    out = []
    for i in range(n):
        l = left[i] if i < len(left) else [(" ", None, None)] * lw
        r = right[i - top] if 0 <= i - top < len(right) else []
        out.append(render(l + [(" ", None, None)] * gap + r))
    return out, lw + gap + rw


def function(fname, lines):
    body = "\n".join(lines)
    assert "'" not in body
    return "%s() {\n  printf '%%s\\n' '%s'\n}\n" % (fname, body)


def main():
    wide, wide_w = compose(["ALEXANDER", "YW"])
    narrow, narrow_w = compose(["ALEX", "ANDER", "YW"])
    print("# shellcheck shell=bash")
    print("# The start.sh banner: Mimir's head above his well beside ALEXANDER YW, truecolour half-block art; TensorFold by Ash Hart under it.")
    print("# Generated by scripts/make_banner.py; `banner` prints the widest form the terminal fits (%d or %d+ columns), else nothing."
          % (wide_w, narrow_w))
    print()
    print(function("_banner_wide", wide))
    print(function("_banner_narrow", narrow))
    print("banner() {")
    print("  [[ -t 1 ]] || return 0                       # a log or a pipe: no art")
    print("  local cols")
    print("  cols=$(tput cols 2>/dev/null || echo 120)")
    print("  if (( cols > %d )); then _banner_wide; elif (( cols > %d )); then _banner_narrow; fi" % (wide_w, narrow_w))
    print("}")


if __name__ == "__main__":
    main()
