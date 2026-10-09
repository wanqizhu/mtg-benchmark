# Historical creator calibration — updated model rules

Four active sources: Casual Spike, Alex Nikolic, LSV + Marshall / Limited Resources,
and Draftsim bot. Original editions are admitted through three calendar days
post Arena launch. Native source facts and labels are preserved. Historical
training uses official FIRST_WEEK PremierDraft GIH, all users and colors,
reported outcomes with at least 500 games, and nonbasic main-set cards.
Each card-set observation has equal weight.

## Requested changes

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
| F | 52.00690% (20) | 50.62165% (20) | 49.26828% (13) |
| D- | 52.75218% (22) | 50.79565% (26) | 51.43784% (1) |
| D | 52.86566% (65) | 52.09870% (72) | 53.21294% (61) |
| D+ | 53.86647% (101) | 53.06636% (157) | 53.59529% (16) |
| C- | 53.86647% (116) | 54.57734% (187) | 53.59529% (27) |
| C | 55.05715% (218) | 55.35578% (200) | 54.48918% (102) |
| C+ | 56.01745% (192) | 56.64991% (221) | 55.53417% (67) |
| C++ | 56.45768% (0) | 57.08437% (0) | 55.53417% (41) |
| B- | 56.89791% (150) | 57.51884% (132) | 55.53417% (49) |
| B | 58.41040% (135) | 58.98822% (85) | 56.24438% (119) |
| B+ | 59.40137% (83) | 60.63965% (60) | 58.75319% (73) |
| A- | 60.25391% (23) | 61.03106% (41) | 60.02972% (21) |
| A | 61.61057% (26) | 62.62489% (46) | 62.10231% (23) |
| A+ | 61.96420% (11) | 65.72842% (11) | 62.75242% (4) |

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
| Casual Spike | build_around offset, pp | -0.62308523 | 27 | 0 |
| Alex Nikolic | build_around offset, pp | -0.39161851 | 22 | 0 |
| Alex Nikolic | synergy offset, pp | -0.32280844 | 52 | 0 |
| LSV + Marshall / Limited Resources | build_around offset, pp | -1.89227389 | 33 | 0 |
| LSV + Marshall / Limited Resources | scenario offset, pp | -0.14739191 | 24 | 0 |
| LSV + Marshall / Limited Resources | build_around conditional grade gap, pp/native step | +0.43705637 | 5 | 5 |
| LSV + Marshall / Limited Resources | scenario conditional grade gap, pp/native step | +0.00000000 | 13 | 0 |

## FRA evaluation

The original October 7 ALL_TIME outcome snapshot remains frozen. The new shared
pool has 266 cards: the verified qualitative sideboard assessment of Refute
Destiny now supplies an F model input under the requested alias. Four other LR
qualitative assessments remain unmapped. The report also compares both policies
on the prior identical 265-card pool.

| Source | Training observations | Historical sets | Pearson r | Spearman rho | MAE, pp | RMSE, pp |
|---|---:|---:|---:|---:|---:|---:|
| Casual Spike | 1162 | 5 | 0.690 | 0.648 | 2.333 | 3.073 |
| Alex Nikolic | 1258 | 5 | 0.778 | 0.700 | 2.128 | 2.720 |
| LSV + Marshall / Limited Resources | 543 | 5 | 0.717 | 0.650 | 2.301 | 2.932 |
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
sideboard merge and regression cap were requested after that evaluation.
Numerical fitting/model selection still use history only. First-week GIH is
noisy and depends on deck, player and cohort composition. FRA's early ALL_TIME
horizon is about 7.6 days; historical windows use the site's FIRST_WEEK.

## Reproduce

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
