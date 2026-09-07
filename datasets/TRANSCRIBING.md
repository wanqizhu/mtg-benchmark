# Transcribing a puzzle into `problem_gold.md`

You are producing **ground-truth board states** for a benchmark. A model will be scored
against these, so a single wrong land count silently corrupts every rollout on that puzzle.
This document exists because the first 20 puzzles were transcribed badly, twice, and every
mistake below is one that actually happened.

Tooling: `python scripts/dataset/scaffold_puzzle.py --help`

---

## 1. The prime directive

> **The puzzle image and the official solution are both ground truth, and they always agree.
> If your transcription makes the solution not work, *you misread the image*.**

Do not "fix" the puzzle. Do not conclude the source is wrong. Go back and re-crop.

This is the single highest-value rule here. Worked example (puzzle 003):

- Thumbnail read said `6x Mountain`. Tight crop said the badge was `5x`.
- The solution spends 7 mana, so 6 lands couldn't be right either.
- Wrong resolutions attempted: "6 Mountains, badge misread", then "5 Mountains, puzzle is buggy".
- Actual answer: the solution pays Brutal Expulsion as **"UURR"** — 2 blue — and blue only
  comes from Wandering Fumarole. So: **5 Mountain + 2 Wandering Fumarole**. There were two
  Fumaroles side by side and the first pass only saw one.

The arithmetic told us exactly what to go look for. Use it that way.

There is exactly one known exception in the corpus, and it is flagged: puzzle **018**'s
published solution was retracted by the publisher and cannot be reconciled with the board
(`"excluded": true` in its `metadata.json`). Assume you are wrong before you assume this.

---

## 2. Source hierarchy

| Source | Authoritative for | Not authoritative for |
|---|---|---|
| **Puzzle image** | What is on the board: which cards, how many, counters, attachments, tapped state, life totals, printed notes | Card rules text (too low-res to trust) |
| **Official solution** (`metadata.json` → `solution_text`) | Cross-checking the board via arithmetic | Zone contents — see §3 |
| **Scryfall** | Exact oracle text, mana cost, P/T, printed loyalty | Anything about *this* board; sometimes the printed type line (see §5) |

**Never transcribe rules text by reading the image.** Identify the card from the art/title,
then pull exact text from Scryfall. Image OCR of a card's text box is the single largest
source of subtle wording errors.

---

## 3. The solution is a *check*, not a *source*

Use the solution to verify counts. Do **not** populate zones from it.

Puzzle 014: the hand was transcribed as "Aviary Mechanic + Cartouche of Ambition" because the
solution mentions a Cartouche. The image plainly showed **2x Rhonas the Indomitable, Aviary
Mechanic, Khenra Charioteer**. The image had been visible the whole time and was overruled by
prose. (Two copies of a legend in hand is perfectly legal — the legend rule is battlefield-only.
Don't let "that looks wrong" talk you out of what you can see.)

---

## 4. Failure modes, all observed

Check each of these explicitly. They are ordered by how often they bit.

1. **Counter dice missed entirely.** Kalitas (003) had a die showing `2` — two +1/+1 counters,
   making it 5/6, transcribed as a plain 3/4. Scan *every* permanent for a die.
2. **Stacked-land counts.** Lands are fanned with an `Nx` badge. Zoom until the digit fills the
   frame, *and* count the card edges behind it. `5x` vs `6x` is a one-pixel difference at
   thumbnail scale. Two different land types can sit adjacent (003: 5 Mountain + 2 Fumarole).
3. **Cards missing from a zone.** 011 was missing Commit // Memory from hand; 015 was missing
   Sparring Mummy from the opponent's board; 013 was missing Glory-Bound Initiate, an equipped
   Hedron Blade, *and* True-Faith Censer. Count the cards in each row of the image and make the
   count match your transcription.
4. **Cards that don't exist.** 010 had a hallucinated Cartouche of Solidarity in hand.
5. **Wholly fabricated boards.** 013 and 014 had opponent boards invented from nothing
   (Oketra/Trueheart Duelist instead of Shimmerscale Drake + 2x Labyrinth Guardian).
6. **Wrong land *types*.** 010: "4x Swamp" was actually 3x Plains + 2x Swamp.
7. **Auras/Equipment read as counters.** 014's Longtusk Cub "+1/+1 counter" was actually a
   **Cartouche of Strength** attached. Look for a card tucked behind/under the creature.
8. **Printed vs current loyalty.** See §5. Several planeswalkers had the *modified* loyalty on
   the type line, which is wrong in both directions.
9. **Deriving a number instead of reading it.** Don't compute the land count from the solution's
   mana total and write that down. Compute it, then go *look*, and reconcile.

---

## 5. Card data rules

**Pull from Scryfall by exact name.** `scaffold_puzzle.py cards "Name"` does this and formats it.

- **Strip all reminder text.** `you get {E}{E} (two energy counters)` → `you get {E}{E}`.
  `Devoid (This card has no color.)` → `Devoid.` Same for Cycling, Embalm, Madness, Crew,
  Escalate, Prowess, Fabricate, Support, Skulk, Emerge, Trample-style reminders — all of it.
- **Type line uses ` - `, not an em dash.** `Creature - Human Wizard`.
- **P/T goes on the type line**, comma-separated: `Creature - Gremlin, 1/2`.
- **Planeswalkers put *printed* loyalty on the type line**: `Legendary Planeswalker - Kiora, loyalty 4`.
  The board's actual loyalty goes in `modifications` as `Current loyalty: 7.` Always include the
  `Current loyalty` line, even when it equals the printed value.
- **Watch for oracle drift on old cards.** Scryfall serves *current* oracle text, which can differ
  from the card as printed in the puzzle's era. Weaver of Currents is `Naga Druid` in Amonkhet but
  `Snake Druid` in current oracle. **The image wins for the type line.** Modern oracle rewrites
  ("this creature" for the card's own name) are fine to keep — the corpus uses current wording.
- **Split / aftermath / DFC cards**: one entry, both faces.
  `Mana cost: {2}{W}{W} // {3}{W}{W}`, `Sorcery // Sorcery`, then
  `Dusk: Destroy all… Dawn: Aftermath. Return all…`
- **Transformed permanents**: use the back face's name and stats, and add
  `Transformed from <front face>.` to `modifications`.
- **Vanilla tokens have no text box**: `- 4x Servo` / `  - Token Artifact Creature - Servo, 1/1`.
- **Basic lands need no entry body** — just `- 3x Island`. Nonbasic lands get a `Land` type line
  and their text box.

---

## 6. Output format

Exactly this template per card, in this order. Omit any part that doesn't apply.

```markdown
- Card Name
  - Mana cost: {2}{B}
  - Legendary Creature - Vampire Warrior, 3/4
  - Entire text box as ONE paragraph. Sentences joined with spaces. Modal spells keep
    their bullets inline: Choose one or both — • Mode A. • Mode B.
  - modifications
    - Has two +1/+1 counters.
```

Rules:

- **The text box is one bullet.** Never split abilities across multiple bullets.
- **The card body is what is *printed*.** No board state leaks into it.
- **`modifications` holds only non-default state**, as sub-bullets.
- **Omit `modifications` entirely** when the card is in its printed state.

### What counts as a modification

| Include | Do **not** include |
|---|---|
| `Current loyalty: 7.` | `Untapped.` — default |
| `Has two +1/+1 counters.` | `Not attached to a creature.` — default for Equipment |
| `Has two brick counters.` | `Already in play, not summoning sick.` — default |
| `Tapped.` | `Currently 5/6.` — derived, see below |
| `Equipped with Hedron Blade.` / `Attached to Kari Zev.` | Anything a global effect grants — see below |
| `Enchanted with Cartouche of Strength.` | |
| `Transformed from Pious Evangel.` | |
| `Enchanting you.` (opponent's Curse) | |

**Never state a computed P/T.** The base is on the type line and the delta is in
`modifications`; `currently 5/6` is redundant and rots if either changes.

**Never propagate a global effect to the cards it affects.** A Gideon emblem
("Creatures you control get +1/+1") is its own battlefield entry. Do **not** write
`Currently 3/3 with the emblem` on every creature. Same for anthem effects
(Rhonas's Monument, Behind the Scenes, Angel of Invention).

**Attachments are recorded on both cards** — `Equipped with X.` on the creature, and
`Attached to Y.` on the Equipment/Aura. Use *Attached to* for the Equipment/Aura side.

### Document skeleton

```markdown
# Puzzle NNN

## Note                          ← only if the image prints notes; verbatim content

## Your Game State

### Life Total                   ← ONLY if the image shows yours; omit otherwise
### Energy                       ← only if an energy counter is shown
### Hand
### Battlefield
### Library                      ← only if the image says something about it
### Graveyard and Exile

## Opponent's Game State

### Life Total
### Battlefield
### Other Zones

## Objective
```

- **Do not invent a life total.** Many puzzles only show the opponent's. If yours isn't
  shown, omit the section — don't write "not specified" or "assume 1".
- **No mana commentary.** Never add "Available mana: {U}{U}{W}" or "these lands could produce
  {4}{R}{R}". The land list already says it.
- **No difficulty/season/assumptions block.** That lives in `metadata.json`.
- Boilerplate, used verbatim:
  - `### Graveyard and Exile` → `- No cards are shown.`
  - `### Other Zones` → `- Assume no relevant opponent hand, graveyard, exile, or library contents.`
  - `## Objective` → `Win this turn.` or `Deal the maximum possible damage this turn.`
- Capture **notes printed on the image** — library contents ("All remaining cards in your library
  are Islands"), opponent-behavior notes (006), prior-turn actions (008). These are load-bearing.

---

## 7. Workflow

```bash
# 0. After a dataset pull: read DIFFICULTY on puzzle.jpg and fix metadata.json.
#    WordPress categories are often leftover from the previous post.

# 1. Look at the whole image, then crop it to pieces. Never work from the thumbnail alone.
python scripts/dataset/scaffold_puzzle.py crop 021

# 2. Read the official solution. Note every number in it.
python scripts/dataset/scaffold_puzzle.py show 021

# 3. Identify cards from crops, then pull exact text.
python scripts/dataset/scaffold_puzzle.py cards "Glorybringer" "Fatal Push"

# 4. Write datasets/mtg/021/problem_gold.md  (init writes a skeleton)
python scripts/dataset/scaffold_puzzle.py init 021

# 5. Machine-check the format.
python scripts/dataset/scaffold_puzzle.py lint 021
```

`lint` catches format drift. It **cannot** catch a wrong land count — only §8 can.

---

## 8. Verification (do not skip)

Reconcile the solution against your transcription arithmetically. On the first 20 puzzles
nearly every one came out **exact**, so an off-by-one is a real signal, not rounding.

1. **Mana.** Sum every cost the solution pays. It should equal your land/rock count —
   and the **colors** must work. This is what caught 003 ("UURR" ⇒ 2 blue sources).
   Watch for lands played from hand (006, 012, 017 have a land in hand that the line uses).
2. **Damage.** Sum the damage dealt. It should equal the opponent's life *exactly*.
3. **P/T claims.** The solution names creature sizes ("a 6/6 Servant", "now an unblockable
   9/8"). Each must equal base + your recorded modifications. This is how 013's Hedron Blade
   and 014's Cartouche were confirmed.
4. **Counter counts.** 009's whole line only works if the counts are exactly right —
   Gearhulk must hit 0/0 on the final trigger, not before.
5. **Blockers.** If the solution says "their best block lets N through", check that the
   creatures you recorded can actually block that way (flying/reach/menace/skulk/defender).
   015's missing Sparring Mummy was caught this way: skulk needs *every* opposing creature
   to have power ≥ 2.

**Be suspicious of a clean bill of health.** Also flag when:

- the solution has a **step that appears skippable** — you probably missed a blocker,
  a counter, or an ability that makes it necessary;
- **many alternative lines** seem to win — the board is probably more constrained than you
  recorded (a missing blocker, a smaller land count);
- the line is **more convoluted than the stated difficulty** implies;
- any number is off **by one**.

Each of those means "go re-crop", not "close enough".

---

## 9. Known source errors

The official solutions are not infallible. Two confirmed:

- **004** — step 6 says "6/6 Druid of the Cowl"; by its own step 4 the Druid is a tapped 5/7 and
  the 6/6 is **Servant of the Conduit**. Corrected in `solution_text`; `solution_text_raw`
  preserves the original scrape.
- **018** — publisher retracted the posted 73-damage answer ("miscalculation in step 8", correct
  max 72), but the corrected line only ever appeared in Patreon comments. A 72 line needs 6 lands;
  the image shows 5 (3x Wandering Fumarole + 2x Spirebluff Canal), capping the real maximum at 71.
  Marked `"excluded": true`, so `load_puzzles()` skips it.

If you find a third, don't silently patch it. Verify hard first (§1), then either fix
`solution_text` and leave `solution_text_raw` intact, or set `"excluded": true` with an
`"excluded_reason"`.

---

## 10. Housekeeping

- `datasets/mtg/` is **gitignored**. Transcriptions are not version-controlled — don't assume
  a mistake is recoverable via git. The `excluded` flag on 018 lives only in that untracked
  metadata, so a fresh rebuild will drop it.
- Work in **one** dataset directory (`datasets/mtg/`, or `$MTG_DATASET_DIR`). An earlier copy
  under `~/Downloads` drifted out of sync and a fix to `004/metadata.json` was silently lost
  for a while. If a second copy exists, diff both trees before and after.
- Never edit `solution_text_raw` — it is the provenance record.
