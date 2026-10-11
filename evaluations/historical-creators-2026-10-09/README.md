# Historical creator calibration — updated model rules

Four active sources: Casual Spike, Alex Nikolic, LSV + Marshall / Limited Resources,
and Draftsim bot. Original editions are admitted through three calendar days
post Arena launch. Native source facts and labels are preserved. Historical
training uses official FIRST_WEEK PremierDraft GIH, all users and colors,
reported outcomes with at least 500 games, and nonbasic main-set cards.
Each card-set observation has equal weight.

## Requested changes

- Each of the three letter-grade creators has exactly one shared context-flag
  coefficient. Explicit base/conditional pairs use only the base bucket, with
  flag 0. Conditional-only grades and single grades with build-around, synergy
  or scenario labels use their supplied grade plus the coefficient, with flag 1.
  Multiple labels activate it once. Tentative alternative grades are excluded.
  Same-grade copies introduced by source assembly do not establish two separate
  ratings. No conditional-grade-gap coefficient is fitted. Where a conditional
  review has an applicability grade plus a narrower scenario grade, the primary
  applicability grade is used (FRA Hexhaven Invigorator: B, not monogreen B+/A−).
- Marc is removed from the active canonical inputs, fit, evaluation and display.
- SB and explicitly sideboard-marked inputs are grouped with F for letter models
  (native 1 for a numeric bot source). There is no separate SB or sideboard-effect
  coefficient. Raw published labels remain visible for provenance.
- For a letter grade with n < 10 real card-set memberships, its final coefficient
  is (n × base coefficient + donor-weighted base coefficients) / 10. Exactly
  10 − n weight is borrowed from closest observed native grades. Equal-distance
  donors share weight equally, capped at actual support; borrowing expands to
  the next distance only after exhausting closer classes. Adjustments are
  simultaneous and use unsmoothed donor coefficients. An unobserved letter
  class borrows all ten weights. Counts remain labelled as real versus borrowed.
- Unordered sparse context effects borrow toward the ordinary-card zero-effect
  baseline: final effect = min(1, n/10) × base effect. This is the explicit
  extension for terms without a nearby ordinal grade class.
- Draftsim uses a monotone regression of its continuous native rating. Candidates
  are linear with two coefficients or cubic B-splines with four through nine.
  Every candidate has fewer than ten total fitted coefficients. Decimal rating
  labels do not receive independent parameters or ten-sample bucket adjustments.

The ten-sample adjustment follows the base historical letter fit, whose
monotone/free family and grade/context penalties are selected using history.
It is applied within every historical validation fold. Large letter classes
receive no ten-sample adjustment; refitting/model selection can change their
base coefficients. Pseudo-sample weight does not create more observed or
independent cards. C++ lies midway between C+ and B−, and split grades retain
original component weights.

## Historical coefficients

Values are final predicted GIH percentages; parentheses show real card-set
memberships. F includes the grouped sideboard inputs.

| Grade | Casual Spike | Alex Nikolic | LSV + Marshall / LR |
|---|---:|---:|---:|
| F | 52.00690% (20) | 50.63960% (20) | 49.26581% (13) |
| D- | 52.75218% (22) | 50.81291% (26) | 51.50322% (1) |
| D | 52.86566% (65) | 52.12515% (72) | 53.33382% (61) |
| D+ | 53.86647% (101) | 53.08473% (157) | 53.62961% (16) |
| C- | 53.86647% (116) | 54.59704% (187) | 53.62961% (27) |
| C | 55.05715% (218) | 55.39071% (200) | 54.49344% (102) |
| C+ | 56.01745% (192) | 56.68349% (221) | 55.56565% (67) |
| C++ | 56.45768% (0) | 57.11061% (0) | 55.56565% (41) |
| B- | 56.89791% (150) | 57.53772% (132) | 55.56565% (49) |
| B | 58.41040% (135) | 59.02025% (85) | 56.25982% (119) |
| B+ | 59.40137% (83) | 60.64768% (60) | 58.84939% (73) |
| A- | 60.25391% (23) | 61.04135% (41) | 60.02512% (21) |
| A | 61.61057% (26) | 62.63394% (46) | 62.18187% (23) |
| A+ | 61.96420% (11) | 65.72981% (11) | 62.79196% (4) |

### Draftsim regression

Selected monotone cubic B-spline regression, with **5 total fitted coefficients**.
Curvature penalty 50; unconstrained smoother trace
2.414805. Historical native-rating domain
[1.0, 4.7]; predictions use the nearest endpoint beyond this range.

| Term | Coefficient |
|---|---:|
| cubic_basis_1 | 46.94643708 |
| cubic_basis_2 | 50.89924660 |
| cubic_basis_3 | 55.10392781 |
| cubic_basis_4 | 59.62331873 |
| cubic_basis_5 | 64.26183714 |

Knot vector: `[1.0, 1.0, 1.0, 1.0, 2.85, 4.7, 4.7, 4.7, 4.7]`.

Prediction is the coefficient-weighted sum of cubic B-spline bases evaluated
at the clipped native rating. Fixed knots add no fitted parameters.

### Context coefficients

| Creator | Term | Final coefficient | Real examples | Borrowed zero-effect weight |
|---|---|---:|---:|---:|
| Casual Spike | Shared context flag, pp | -0.62308523 | 27 | 0 |
| Alex Nikolic | Shared context flag, pp | -0.75189956 | 74 | 0 |
| LSV + Marshall / Limited Resources | Shared context flag, pp | -1.86937656 | 39 | 0 |

## FRA evaluation

The original October 7 ALL_TIME outcome snapshot remains frozen. The shared
pool has 266 cards: the verified qualitative sideboard assessment of Refute
Destiny now supplies an F model input under the requested alias. Four other LR
qualitative assessments remain unmapped. The report also compares both policies
on the prior identical 265-card pool.

| Source | Training observations | Historical sets | Pearson r | Spearman rho | MAE, pp | RMSE, pp |
|---|---:|---:|---:|---:|---:|---:|
| Casual Spike | 1162 | 5 | 0.690 | 0.648 | 2.333 | 3.073 |
| Alex Nikolic | 1258 | 5 | 0.778 | 0.701 | 2.139 | 2.716 |
| LSV + Marshall / Limited Resources | 543 | 5 | 0.710 | 0.644 | 2.295 | 2.958 |
| Draftsim bot | 443 | 2 | 0.606 | 0.565 | 2.557 | 3.354 |

## Validation and limits

Models and penalties are chosen by historical purged leave-one-set-out RMSE.
Every historical training printing matching a held-out scoring-fold card name
is removed. These are tuning scores, not nested unbiased generalization scores.
After selection, refit all admitted history and lock by hash before evaluation
reads FRA outcomes. A guarded repeat fit blocks every FRA file and the current
results folder; coefficients and model choices reproduce exactly.

The new fit has 3,406 eligible observations from 17 admitted source/set editions.
Casual and Alex each contribute five version-verified individual lists. LR has
five sets with partial caption reconstruction outside the full Hobbit
transcription; Avatar contributes CU only. The isolated captured LR D− is
S.H.I.E.L.D. Helicarrier from Marvel. This is not a count of every grade spoken
in the complete reviews. The bot has only two provisionally admitted editions
(HOB/SOS). HOB's secondary snapshots self-report August 11 and August 14 and
agree on all 188 bot ratings; those dates lack independent archive verification.
SOS is a current public asset whose publisher declares April 15 as its latest
update, before the April 21 Arena launch. No independent prelaunch copy or edit
history was recovered, so it could have changed without that date changing.
The bot's numerical results are conditional on these source-timing assumptions.
Lists with explicitly late update dates are excluded. One-set source-strength
sensitivities are available; neither bot edition has the timing assurance of
17Lands' version history.
FRA's bot inputs have the same limitation: the source-activity ledger records
September 25, but the saved collector bundle was generated October 6, after the
September 29 Arena launch. No independent prelaunch bot copy or edit history was
recovered. Its FRA evaluation is therefore provisional too; excluding FRA
outcomes from historical fitting does not verify the input ratings' timing.
No media was played during research.

This is retrospective analysis. The earlier conditional-gap family was added
after inspection of an initial FRA evaluation. The current support rule,
sideboard merge, regression cap and shared context-flag encoding were requested
after that evaluation.
Numerical fitting/model selection still use history only. First-week GIH is
noisy and depends on deck, player and cohort composition. FRA's early ALL_TIME
horizon is about 7.6 days; historical windows use the site's FIRST_WEEK.

## Reproduce

### Benchmark website tab

The `Draft Predictions` tab combines the six original AI percentage forecasts
with these four historical creator models. Its initial score table and scatter
plot use the identical 266-card pool. The complete expandable table includes
295 cards, the ten predictions, and creators' published raw grades. Native
creator grades are converted only by the retained historical fits; the AI
forecasts use their original values without FRA recalibration.

Display names identify the original `gpt-6` submission as **GPT-6 Sol**,
confirmed by its September 24 run metadata (`gpt-6-sol`, xhigh). The original
source ID, commit and numerical forecasts are retained. Alex Nikolic is labeled
with his publication, **Limited Level-Ups**; LSV and Marshall remain the separate
Limited Resources source.

`inputs/card_images.json` contains Scryfall image URLs matched to all 295
canonical print URLs from the retained raw Scryfall metadata. Images load on
card click; their URLs are metadata, and no image files are vendored.

The exporter checks the shared outcome hash, card coverage, and equality with
the existing creator scores. It writes a compact analyzed JSON and a CSV;
live response envelopes remain local. The static assets are included in
`harness/site_static`, and the site generator preserves the exported data when
puzzle scores are updated.

```sh
python3 export_benchmark_tab.py \
  --model-data ../fra-week-one-2026-10-07/results/dashboard_data.json \
  --output /path/to/mtg-benchmark-site/data/draft-predictions.json
```

Open the site at `/#/draft-predictions`. The plot shows predicted versus observed
GIH, with prediction on x and observation on y. Hover shows card rates,
differences and samples; clicking opens its Scryfall image and prediction stats.
The full table defaults to descending observed GIH. Source names are colored
by AI model or content creator. Unused grades with no historical observations
are omitted from the fitted-coefficient display; their fallback mappings remain
in the model artifact, and the predictions are unchanged.
Methodology includes the full shared prediction prompt. Data collection,
fitting and final coefficients are grouped under Content Creator Historical
Data Details. All expansions and the full card table are collapsed initially.

### Historical fits

Dependencies: numpy, pandas, scipy, plotly, markdown. From this directory:

```sh
python3 historical_creators.py fit --truth-dir /path/to/historical-truth
python3 historical_creators.py evaluate --truth-dir /path/to/historical-truth \
  --fra-truth /path/to/original-17lands_fra_all_time.json
python3 verify_historical.py --truth-dir /path/to/historical-truth \
  --source-scratch /path/to/historical-calibration
python3 present_historical.py --destination /path/to/display-folder
```

Historical truth requires `<SET>_FIRST_WEEK.json` and `<SET>.json` for HOB, MSH,
SOS, TMT, ECL and TLA. Each JSON envelope holds its API response under `data`.
Endpoints use `https://www.17lands.com/api/card_data?expansion=<SET>&event_type=PremierDraft&time_period=FIRST_WEEK`
and the same URL with `time_period=ALL_TIME`. The FRA envelope must match the
hash in the sibling `fra-week-one-2026-10-07/inputs/ground_truth_manifest.json`;
a fresh download cannot reproduce that snapshot. Live envelopes stay local.

Source facts in `inputs/` are canonical refit inputs. `collection/` documents
the original scratch research and manual attribution checks; it is not a fresh
web scrape that silently replaces the frozen checked inputs. Local original
scratch is `work/historical-calibration` relative to the task workspace.

Generated `results/` remain Gitignored. The offline dashboard has plots,
card hover/click details, complete 295 × 4 raw/mapped tables, source audit,
historical coefficients and exact borrowed weights. Downloads include
`model_coefficients.csv`, `borrowed_sample_weights.csv`, `historical_models.json`,
`historical_report.md`, CV candidates/predictions, sensitivity scores and
`validation.json`. Bot grade-grid values in `grade_mappings.csv` are curve
evaluations, not extra coefficients. The source ZIP includes sibling catalog
paths and derived results, with no live API envelopes. The optional prior-score
summary enables the fixed 265-card comparison panel.
