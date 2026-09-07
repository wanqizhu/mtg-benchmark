"""Scaffolding for hand-transcribing puzzles into problem_gold.md.

Read datasets/TRANSCRIBING.md first. The rules that matter are there; this file
only automates the mechanical parts.

    crop  <id>              slice puzzle.jpg into upscaled crops for close reading
    show  <id>              print metadata + official solution + current gold
    cards <name> [name ...] fetch Scryfall and emit ready-to-paste entry blocks
    init  <id>              write a skeleton problem_gold.md
    lint  [id ...]          check format compliance (default: all puzzles)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.mtg.dataset import dataset_dir  # noqa: E402

SCRYFALL = "https://api.scryfall.com/cards/named?exact="

BOILERPLATE_GRAVEYARD = "- No cards are shown."
BOILERPLATE_OTHER = "- Assume no relevant opponent hand, graveyard, exile, or library contents."

REQUIRED_SECTIONS = [
    "## Your Game State",
    "### Hand",
    "### Battlefield",
    "## Opponent's Game State",
    "## Objective",
]

KNOWN_SECTIONS = {
    "## Note",
    "## Your Game State",
    "## Opponent's Game State",
    "## Objective",
    "### Life Total",
    "### Energy",
    "### Hand",
    "### Battlefield",
    "### Library",
    "### Graveyard and Exile",
    "### Other Zones",
}

# Type-line heads that legitimately start a card's type bullet.
TYPE_HEAD = re.compile(
    r"^(Legendary |Basic |Token |Snow )*"
    r"(Artifact|Creature|Enchantment|Instant|Sorcery|Land|Planeswalker|Emblem|Battle|Kindred|Tribal)"
)


def puzzle_dir(puzzle_id: str) -> Path:
    path = dataset_dir() / puzzle_id
    if not path.is_dir():
        sys.exit(f"no such puzzle: {path}")
    return path


def image_path(pdir: Path) -> Path:
    for name in ("puzzle.png", "puzzle.jpg"):
        if (pdir / name).exists():
            return pdir / name
    sys.exit(f"no puzzle image in {pdir}")


# --------------------------------------------------------------------------- crop


def cmd_crop(args: argparse.Namespace) -> None:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("pillow required: pip install pillow")

    pdir = puzzle_dir(args.puzzle_id)
    src = image_path(pdir)
    out = Path(args.outdir) if args.outdir else Path("/tmp") / f"puzzle-{args.puzzle_id}-crops"
    out.mkdir(parents=True, exist_ok=True)

    im = Image.open(src)
    w, h = im.size

    def save(name: str, box: tuple[int, int, int, int], scale: float = 2.0) -> None:
        crop = im.crop(box)
        cw, ch = crop.size
        crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS).save(out / f"{name}.png")

    save("00-full", (0, 0, w, h), 1.0)
    # Footer strip: DIFFICULTY / set name. WP metadata is often stale — read this.
    save("05-footer", (0, int(h * 0.88), w, h), 2.0)

    # Possibility Storm layout: opponent row on top, your board in the middle,
    # your hand along the bottom. Bands overlap so nothing falls in a seam.
    bands = {
        "10-opponent": (0.00, 0.26),
        "20-board": (0.30, 0.68),
        "30-hand": (0.66, 1.00),
    }
    for name, (top, bottom) in bands.items():
        save(name, (0, int(h * top), w, int(h * bottom)))
        # Halves of each band, for reading card text and count badges.
        save(f"{name}-L", (0, int(h * top), int(w * 0.55), int(h * bottom)), 2.5)
        save(f"{name}-R", (int(w * 0.45), int(h * top), w, int(h * bottom)), 2.5)

    # Column tiles across the board row: where count badges and counter dice live.
    for i in range(4):
        x0, x1 = int(w * (0.04 + 0.24 * i)), int(w * (0.30 + 0.24 * i))
        save(f"40-board-col{i}", (x0, int(h * 0.30), min(x1, w), int(h * 0.70)), 2.5)

    print(f"wrote {len(list(out.glob('*.png')))} crops to {out}")
    print(textwrap.dedent("""
        Now actually look at them. Specifically:
          - 05-footer: DIFFICULTY on the image; fix metadata.json if WP was stale
          - every permanent: is there a die on it? (counters)
          - every land stack: read the `Nx` badge AND count card edges
          - every creature: is a card tucked behind it? (aura/equipment)
          - count cards per row; make your transcription match
    """).strip())


# -------------------------------------------------------------------------- show


def cmd_show(args: argparse.Namespace) -> None:
    pdir = puzzle_dir(args.puzzle_id)
    meta = json.loads((pdir / "metadata.json").read_text(encoding="utf-8"))

    print(f"=== puzzle {args.puzzle_id} ===")
    for key in ("difficulty", "seasons", "solution_url", "excluded", "excluded_reason"):
        if key in meta:
            print(f"{key}: {meta[key]}")
    print(f"\n=== official solution ===\n{meta.get('solution_text', '(none)')}")

    gold = pdir / "problem_gold.md"
    if gold.exists():
        print(f"\n=== current problem_gold.md ===\n{gold.read_text(encoding='utf-8')}")
    else:
        print("\n(no problem_gold.md yet)")

    print(textwrap.dedent("""
        === verify (datasets/TRANSCRIBING.md §8) ===
          The solution is a check, not a source. Do not populate zones from it.
          1. sum every mana cost paid -> must equal your lands, colors included
          2. sum damage dealt -> must equal opponent life exactly
          3. every P/T the solution names -> base + your modifications
          4. can the blockers you recorded actually block as described?
        Off by one => you misread the image. Go re-crop.
    """).strip())


# ------------------------------------------------------------------------- cards


def strip_reminder(text: str) -> str:
    """Drop parenthetical reminder text. Scryfall ships it; the corpus doesn't."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\s*\([^()]*\)", "", text)
    return text.strip()


def join_oracle(text: str) -> str:
    """Flatten oracle text to one paragraph.

    Scryfall separates abilities with newlines and leans on the line break instead of
    punctuation, so keyword lines arrive bare ("Flying, haste", "Cycling {2}",
    "Aftermath"). The corpus terminates every ability, so add a period where the line
    doesn't already end in punctuation.
    """
    out = []
    for raw in text.split("\n"):
        line = strip_reminder(raw)
        if not line:
            continue
        if line[-1] not in '.!?"\'—–:':
            line += "."
        out.append(line)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def format_entry(data: dict) -> str:
    """Render one Scryfall payload as a problem_gold.md entry block."""
    faces = data.get("card_faces")
    layout = data.get("layout", "")

    if faces and layout in {"split", "aftermath", "modal_dfc", "transform", "flip"}:
        cost = " // ".join(f.get("mana_cost", "") or "—" for f in faces)
        types = " // ".join(f["type_line"].replace("—", "-") for f in faces)
        body = " ".join(
            f"{f['name']}: {join_oracle(f.get('oracle_text', ''))}" for f in faces
        )
    else:
        cost = data.get("mana_cost", "")
        types = data["type_line"].replace("—", "-")
        if data.get("power") is not None:
            types += f", {data['power']}/{data['toughness']}"
        elif data.get("loyalty") is not None:
            types += f", loyalty {data['loyalty']}"
        body = join_oracle(data.get("oracle_text", ""))

    body = re.sub(r"\s+", " ", body).strip()

    lines = [f"- {data['name']}"]
    if cost and cost != "—":
        lines.append(f"  - Mana cost: {cost}")
    lines.append(f"  - {types}")
    if body:
        lines.append(f"  - {body}")
    return "\n".join(lines)


def fetch_json(url: str) -> dict:
    """Scryfall fetch. Uses certifi when present; falls back to curl, since a bare
    python.org interpreter on macOS often has no usable CA bundle."""
    try:
        import ssl

        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=20, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        import subprocess

        out = subprocess.run(
            ["curl", "-sSL", "--fail", url], capture_output=True, text=True, timeout=30
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or f"curl exited {out.returncode}") from None
        return json.loads(out.stdout)


def cmd_cards(args: argparse.Namespace) -> None:
    for name in args.names:
        url = SCRYFALL + urllib.parse.quote(name)
        try:
            data = fetch_json(url)
        except Exception as exc:  # noqa: BLE001
            print(f"- {name}\n  !! lookup failed: {exc}\n")
            continue

        print(format_entry(data))
        if data.get("loyalty") is not None:
            print("  - modifications\n    - Current loyalty: <read from image>.")
        print()

    print(
        "# reminders: printed loyalty stays on the type line; put board loyalty under\n"
        "# modifications. Old cards can have a different printed type line than current\n"
        "# oracle (Weaver of Currents: Naga vs Snake Druid) -- the IMAGE wins."
    )


# -------------------------------------------------------------------------- init


SKELETON = """# Puzzle {pid}

## Your Game State

### Hand

- TODO

### Battlefield

- TODO

### Graveyard and Exile

{graveyard}

## Opponent's Game State

### Life Total

- Opponent's life total: TODO.

### Battlefield

- TODO

### Other Zones

{other}

## Objective

{objective}
"""


def cmd_init(args: argparse.Namespace) -> None:
    pdir = puzzle_dir(args.puzzle_id)
    gold = pdir / "problem_gold.md"
    if gold.exists() and not args.force:
        sys.exit(f"{gold} exists (use --force to overwrite)")

    # Deliberately not guessed: cleaned solution_text has no preamble, so keyword
    # sniffing silently mislabels max-damage puzzles as "Win this turn."
    objective = "TODO -- one of: Win this turn. | Deal the maximum possible damage this turn."

    gold.write_text(
        SKELETON.format(
            pid=args.puzzle_id,
            graveyard=BOILERPLATE_GRAVEYARD,
            other=BOILERPLATE_OTHER,
            objective=objective,
        ),
        encoding="utf-8",
    )
    print(f"wrote skeleton {gold}")
    print("set the objective from the image prompt ('Can you win this turn?' vs")
    print("'How much damage can you deal?') -- do not infer it from the solution.")
    print("add ### Life Total / ### Energy / ### Library / ## Note only if the image shows them.")


# -------------------------------------------------------------------------- lint


def lint_file(path: Path, puzzle_id: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    problems: list[str] = []

    def err(i: int | None, msg: str) -> None:
        problems.append(f"{path.parent.name}" + (f":{i + 1}" if i is not None else "") + f"  {msg}")

    if not lines or lines[0] != f"# Puzzle {puzzle_id}":
        err(0, f"first line must be '# Puzzle {puzzle_id}'")
    for section in REQUIRED_SECTIONS:
        if section not in lines:
            err(None, f"missing required section {section!r}")
    for i, line in enumerate(lines):
        if line.startswith("#") and line not in KNOWN_SECTIONS and not line.startswith("# Puzzle"):
            err(i, f"unexpected section {line!r}")

    # Banned leftovers from earlier format revisions.
    for i, line in enumerate(lines):
        if re.search(r"\bcurrently \d+/\d+", line):
            err(i, "states a computed P/T; keep base on type line, delta in modifications")
        if re.search(r"base \d+/\d+ with", line):
            err(i, "old 'base X/Y with ...' form; split into type line + modifications")
        if "current loyalty" in line.lower() and "- modifications" not in line and line.strip().startswith("- "):
            if not line.strip().startswith("- Current loyalty"):
                err(i, "current loyalty on type line; printed loyalty belongs there")
        if re.match(r"^  - \(.*\)$", line):
            err(i, "old parenthetical modification; use a 'modifications' sub-list")
        # "could produce" appears in real oracle text (Naga Vitalist), so match the
        # commentary heading only.
        if re.search(r"\bAvailable mana\b|^### Mana Available", line):
            err(i, "mana commentary; the land list is sufficient")
        if re.search(r"emblem\.\)|with the emblem", line):
            err(i, "global buff propagated onto an affected card; keep it on the emblem entry")

    # Per-entry structure.
    i = 0
    while i < len(lines):
        if re.match(r"^- \S", lines[i]) and not lines[i].startswith("- No cards") \
                and not lines[i].startswith("- Assume") and not lines[i].startswith("- Opponent's") \
                and not lines[i].startswith("- You have") and not lines[i].startswith("- Your life") \
                and not lines[i].startswith("- All remaining") and not lines[i].startswith("- Empty") \
                and not lines[i].startswith("- The "):
            start = i
            subs: list[tuple[int, str]] = []
            i += 1
            while i < len(lines) and re.match(r"^ {2,}- ", lines[i]):
                subs.append((i, lines[i]))
                i += 1
            if subs:
                idx = next((k for k, (_, s) in enumerate(subs) if s.strip() == "- modifications"), None)
                body = subs if idx is None else subs[:idx]
                tail = [] if idx is None else subs[idx + 1:]

                p = 0
                if body and body[0][1].strip().startswith("- Mana cost:"):
                    p = 1
                if p < len(body):
                    tline = body[p][1].strip()[2:]
                    if not TYPE_HEAD.match(tline):
                        err(body[p][0], f"expected a type line, got {tline[:48]!r}")
                rest = body[p + 1:]
                if len(rest) > 1:
                    err(rest[1][0], "text box split across bullets; join into ONE paragraph")
                for j, s in rest:
                    if re.search(r"\([^)]*\)", s):
                        err(j, "possible reminder text left in card body")
                for j, s in tail:
                    if not re.match(r"^ {4}- ", s):
                        err(j, "modifications entries must be indented 4 spaces")
                for j, s in tail:
                    low = s.lower()
                    if any(k in low for k in ("untapped", "not summoning sick", "not currently attached",
                                              "not attached", "already in play")):
                        err(j, "default state listed as a modification; delete it")
            continue
        i += 1

    if not text.endswith("\n"):
        err(None, "file should end with a newline")
    return problems


def cmd_lint(args: argparse.Namespace) -> None:
    root = dataset_dir()
    ids = args.puzzle_ids or sorted(
        d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit()
    )
    total, checked = 0, 0
    for pid in ids:
        gold = root / pid / "problem_gold.md"
        if not gold.exists():
            continue
        checked += 1
        problems = lint_file(gold, pid)
        total += len(problems)
        for p in problems:
            print(p)
    print(f"\nlinted {checked} puzzle(s); {total} problem(s)")
    if total:
        sys.exit(1)


# -------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("crop", help="slice the puzzle image into upscaled crops")
    p.add_argument("puzzle_id")
    p.add_argument("--outdir")
    p.set_defaults(func=cmd_crop)

    p = sub.add_parser("show", help="print metadata, solution, and current gold")
    p.add_argument("puzzle_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("cards", help="fetch Scryfall and emit entry blocks")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=cmd_cards)

    p = sub.add_parser("init", help="write a skeleton problem_gold.md")
    p.add_argument("puzzle_id")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("lint", help="check format compliance")
    p.add_argument("puzzle_ids", nargs="*")
    p.set_defaults(func=cmd_lint)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
