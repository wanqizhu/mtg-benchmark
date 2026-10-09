# Historical creator calibration, October 9, 2026

Fit each creator's original native grades to historical Premier Draft GIH win
rates, then transfer that mapping to Reality Fracture (FRA). This branch contains
the code and curated prediction facts. Live response envelopes and generated
outcome tables stay local under the repository's ignored `results/` directory.

## Result

The identical 265-card FRA comparison uses the frozen October 7 snapshot from
the earlier evaluation, not a new October 9 download.

| Source | Historical sets | Eligible training observations | Pearson r | Spearman rho | MAE, percentage points | RMSE, percentage points |
|---|---:|---:|---:|---:|---:|---:|
| Marc Anderson | 5 | 1,259 | 0.777 | 0.727 | 2.12 | 2.75 |
| Alex Nikolic | 5 | 1,258 | 0.771 | 0.697 | 2.15 | 2.76 |
| Casual Spike | 5 | 1,162 | 0.687 | 0.644 | 2.34 | 3.08 |
| LSV + Marshall / Limited Resources | 5, partial | 543 | 0.680 | 0.613 | 2.37 | 3.08 |
| Draftsim bot | 2 | 443 | 0.617 | 0.563 | 2.54 | 3.31 |

Marc and Alex's mean absolute errors differ by only 0.028 points. A descriptive
paired card bootstrap includes zero; these results do not establish a clear
winner between them. The all-card output retains 295 canonical cards and five
creator columns, with missing grades left blank. Four creators score 270
reported main-set cards; LR scores 265.

## Original-source audit

Admission uses an original edition published no later than three calendar days
after Arena release. Later edits are excluded using version history whenever
available. Up to five recent sets are included per source, without selecting
sources for their outcome performance.

| Source | Admitted historical sets | Evidence and limits |
|---|---|---|
| Casual Spike | HOB, MSH, SOS, TMT, ECL | Original 17Lands versions linked from dated authored posts; original flags frozen too. TMT has one ungraded main card. |
| Alex Nikolic | MSH, SOS, TMT, ECL, TLA | Separate original 17Lands versions. The joint Alex/Marc Hobbit list is excluded. TMT has one ungraded main card. |
| Marc Anderson | MSH, SOS, TMT, ECL, TLA | Separate original 17Lands versions. The joint Hobbit list is excluded. |
| LR | HOB, MSH, SOS, ECL, TLA | HOB has a full dated secondary transcription. Other sets have partial, manually checked caption reconstructions. TLA has CU only. This is a joint LSV/Marshall source. |
| Draftsim bot | HOB, SOS | HOB has early self-dated collector files; SOS has a publisher latest-update date preceding release. These guarantees are weaker than 17Lands history. Later revised bot tables are excluded. |

Set codes: HOB = The Hobbit; MSH = Marvel Superheroes; SOS = Secrets of
Strixhaven; TMT = Teenage Mutant Ninja Turtles; ECL = Lorwyn Eclipsed;
TLA = Avatar: The Last Airbender.

LR rares were available within the permitted window: [HOB August 13](https://lrcast.com/limited-resources-866-the-hobbit-set-review-rare-and-mythic-rare/),
[MSH June 25](https://lrcast.com/limited-resources-859-marvel-superheroes-set-review-rare-and-mythic-rare/),
and [ECL January 22](https://lrcast.com/limited-resources-837-lorwyn-eclipsed-set-review-rare-and-mythic-rare/)
were each two days after Arena launch; [SOS April 21](https://lrcast.com/limited-resources-850-secrets-of-strixhaven-set-review-rare-and-mythic-rare/)
was release day. TLA's rare review was four days after launch and is excluded.
The dated audio pages establish timing; caption text supplies card evidence.
No media was played during collection.

The canonical input files preserve every retained raw grade, conditional grade,
source timestamp, original-version identifier and available hash. The audit
also records rejected editions and ambiguous spoken statements. Collection
helpers document the scratch workflow; they require the original workspace
scratch layout. For refitting, use the curated `inputs/historical_ratings.json`
directly. The helpers are not intended to silently recreate manually checked
facts from fresh web content.

## Model

The historical target is 17Lands' official `FIRST_WEEK` PremierDraft
`ever_drawn_win_rate`, all tracked users and colors. Training uses nonbasic
main-set cards with a reported GIH and at least 500 GIH games. Each card-set
observation has equal weight. There are 4,755 captured source rows and 4,665
eligible training observations.

Each native letter or decimal grade receives a fitted win-rate parameter.
Equivalent numeric strings share a bucket; C++ stays between C+ and B−. Split
grades average their component bucket mappings. SB is an unordered bucket.
Observed context types receive separate offsets. Explicit conditional-only
letters can supply the input when an ordinary grade is absent. Qualitative
assessments without a recoverable letter or number remain unmapped.

Conditional combinations such as build-around B share the native grade curve
and context effects. They are not given separate unrelated parameters: almost
all such combinations still have fewer than ten observations. All captured
combinations and their actual counts remain available in the conditional table.

The objective is equal-card squared error plus penalties on adjacent grade
differences, SB's distance from the lowest supported ordinary bucket, and
context coefficients. Model selection compares monotone/free curves; grade
penalties 0, 2, 10, 50; context penalties 2, 10, 50; and conditional-grade-gap
terms on/off. Purged leave-one-set-out selection removes every training card
name appearing in the held-out scoring fold. These CV results are tuning scores, not
nested unbiased estimates.

| Source | Selected grade penalty | Selected context penalty | Conditional-gap terms |
|---|---:|---:|---|
| Casual Spike | 10 | 50 | Off |
| Alex | 10 | 50 | Off |
| Marc | 10 | 50 | Off |
| LR | 2 | 10 | On |
| Bot | 0 | 50, unused | Off |

All selected curves are monotone. Every observed bucket is retained, including
buckets with fewer than ten cards. Smoothing borrows strength; it does not
manufacture ten observations. Omit-one-set mapping ranges show sensitivity,
not confidence intervals. The exact coefficients and counts are generated in
`historical_models.json`, `grade_mappings.csv`, `context_coefficients.csv`, and
the raw/conditional bucket count tables.

LR's conditional build-around gap is fitted from only five nonzero examples.
Its contribution is provisional. The gap family was added after inspecting
the initial flag-only FRA evaluation. Selection and fitting still use history
only, and the initial results are retained. This is a retrospective transfer
analysis, not an analyst-blind or preregistered forecast. Casual's fifth set
was recovered later through source research and admitted on provenance.

FRA outcomes are never read by the fitting stage. Its saved model is locked by
hash before evaluation. Historical printings of FRA cards are allowed as past
information; a sensitivity excludes them. Other sensitivity runs use mature
ALL_TIME historical targets, grade-only terms, no postlaunch LR rare episodes,
and each individual bot set. Hyperparameters remain fixed in sensitivity runs.

## Reproduce locally

Python dependencies: numpy, pandas, scipy, plotly and markdown. The scripts use
paths relative to their own location. From this analysis directory:

```sh
python3 historical_creators.py fit --truth-dir /path/to/historical-truth
python3 historical_creators.py evaluate --truth-dir /path/to/historical-truth \
  --fra-truth /path/to/original-17lands_fra_all_time.json
python3 present_historical.py --destination /path/to/display-folder
```

`historical-truth` must contain `<SET>_FIRST_WEEK.json` and `<SET>.json` for all
six admitted set codes. Each is a JSON envelope with the API result in `data`.
Endpoints:

```text
https://www.17lands.com/api/card_data?expansion=<SET>&event_type=PremierDraft&time_period=FIRST_WEEK
https://www.17lands.com/api/card_data?expansion=<SET>&event_type=PremierDraft&time_period=ALL_TIME
```

The FRA file must match the original SHA-256 in the sibling
`fra-week-one-2026-10-07/inputs/ground_truth_manifest.json`; a fresh download
cannot reproduce that frozen snapshot. The source ZIP preserves this sibling
layout and includes the canonical catalog, curated inputs, code, and published
derived results, including the initial flag-only comparison. Live API response
envelopes are not included. A fresh clone also needs the preserved initial
flag-only results to generate its comparison panel.

The original local scratch inputs are under `work/historical-calibration`
relative to the task workspace. Verification:

```sh
python3 verify_historical.py --truth-dir /path/to/historical-truth \
  --source-scratch /path/to/historical-calibration
```

This independently refits with all FRA files forbidden, checks original tier
history replay, verifies percentages/monotonicity, checks the 295 × 5 export,
recomputes shared-card metrics, and confirms FIRST_WEEK responses differ from
mature ALL_TIME data. The completed audit is in `results/validation.json`.

## Read the results

`historical_creators.html` is a standalone offline dashboard: no remote scripts,
video, audio, or tracking. Scatter hover/click includes native grades, mapped
rates, sample sizes and full card rules; filters include creator, rarity, name
and the shared comparison pool. It also has grade curves, color residuals,
all-card raw/mapped tables, source audits, sparse counts and exact models.

`historical_report.md` explains the methods and largest errors. Common errors
include underpredicting blue, overpredicting red, and compressing Ajani's
extreme observed rate into the historical A/A+ average. High grades alone do
not distinguish exceptional bombs within the same grade bucket.

Limits: partial LR transcription can bias the historical sample; the bot has
only two weaker dated editions; sparse contextual and extreme grade buckets
remain; first-week GIH is noisy and affected by deck/player context. FRA's
original early ALL_TIME horizon is about 7.6 days, slightly longer than the
historical FIRST_WEEK window.
