Place the MTG puzzles dataset in `datasets/mtg/`.

That directory is gitignored. It should contain `common/comp_rules.txt`, `common/clarifications.txt`, and numbered puzzle folders (`001/`, `002/`, …).

Override the location with `MTG_DATASET_DIR` if needed.

## Rebuild from sources

Puzzle catalog and images come from the public Possibility Storm WordPress API. Official solutions are Patreon-only.

Patreon auth is **your existing browser session**, not a password. Log into [patreon.com](https://www.patreon.com) in Chrome (or another supported browser), then:

```bash
pip install gallery-dl
python scripts/dataset/build_dataset.py --browser chrome
```

`gallery-dl --cookies-from-browser chrome` reads the `session_id` cookie from the local browser profile. `--user-agent browser` sends a real browser UA. Close Chrome if cookie unlock fails.

After a pull, **check difficulty on each new `puzzle.jpg` footer** (`DIFFICULTY: UNCOMMON/RARE/MYTHIC`) and fix `metadata.json` if the WordPress category is leftover from the previous post. Season is taken from the title when it looks like `302: Marvel Super Heroes #1`; otherwise from WP tags, which can also be stale.

Then transcribe `problem_gold.md` from the image (not from the solution). See `datasets/TRANSCRIBING.md`.

The older bulk dump (every Patreon post, not just numbered puzzles) is:

```bash
scripts/dataset/download_patreon.sh chrome
```

That writes to `datasets/patreon-raw/` (also gitignored). After a fresh `build_dataset.py` run, apply `scripts/clean_solution_text.py` to split newsletter dumps into `solution_text_raw` / cleaned `solution_text`.

Optional hand-written parser overrides live in `datasets/manual_solutions.json` (gitignored). If that file is missing, the cleaner uses the parser only.
