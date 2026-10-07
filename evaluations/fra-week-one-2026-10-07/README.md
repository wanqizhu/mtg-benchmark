# FRA early-format evaluation

Local analysis on `gpt-6-fra-week-one-evaluation`. Original predictions are retained at the immutable submitted commits from PRs #8–#13. No forecast is modified.

## Reproduce

Use Python 3.11+ with the packages in `requirements.txt`:

```sh
python evaluate.py --truth inputs/local/17lands_fra_all_time.json
```

The supplied response must match `inputs/ground_truth_manifest.json`. It is the October 7 ALL_TIME snapshot requested by the user, rather than an exact seven-calendar-day slice. Suppressed rates stay missing.

Live 17Lands data and generated results stay local. `inputs/local/` and `results/` are gitignored. Results and the offline interactive dashboard are also copied to the task's `outputs/fra-week-one/` directory for review. This branch contains the reproducible code, frozen model submissions, native creator grades with attribution, card metadata, and source manifests. Before distributing results, consult [17Lands' usage guidelines](https://www.17lands.com/usage_guidelines) and its data-use notice. The public FRA game-data dump was unavailable at analysis time.

## Outputs

| File in results/ | Purpose |
| --- | --- |
| dashboard.html | Offline scatterplots, source and sample filters, card rules on hover/click, matched leaderboards, conversion audit |
| report.md / report.html | Findings, methodology, source dates, uncertainty, model and creator mistakes |
| cards_and_predictions.csv | One row per canonical card, all raw forecasts and grade indexes, native grades, held-out conversions, metadata and observed GIH |
| predictions_long.csv | One row per card/source, with explicit units, original value, interpreted prediction, conversion, observed rate, sample count and scoring eligibility |
| scores.csv | Raw model and converted creator scores for available, matched, sample-threshold and Special Guest pools |
| paired_comparisons.csv | Paired 95% card-bootstrap intervals for MAE differences |
| coverage.csv | Submission coverage and genuinely context-only omissions |
| calibration.csv | Grade means, counts, final descriptive isotonic mapping and out-of-fold averages |
| group_errors.csv | Color, rarity, mana, and mechanic errors (groups overlap) |
| mistakes.csv | Top-ten overpredictions and underpredictions for each source |
| validation.json | Checksums, pool counts, fold audit, parsing and metric checks |
| dashboard_data.json | Embedded dashboard observations, metadata and provenance |

Win rates use a 0–100 percentage scale. An error of 2 means two percentage points. `predicted_wr` in the long table is the unchanged original percentage for models, and the held-out conversion for creators. `calibrated_wr` applies the same conversion procedure to every predictor. Creator `original_value` is an ordering index, never a percentage forecast. Missing forecasts or suppressed observed rates remain blank. `scored_main` marks eligibility for that source's available main-set comparison; matched pools are more restrictive.

Conversions use increasing isotonic regression in five shared card folds repeated twenty times. Every scored card is absent from each mapping that predicts it. A full-data monotone mapping appears only as a descriptive audit or to predict supplemental Special Guests, which never train a mapping. Ordinary grades with conditional alternatives are retained at their ordinary grade. Conditional-only and sideboard-only grades remain unscored.

## Inputs and provenance

`inputs/model_sources.json` identifies each PR, exact commit and original repository path. `inputs/models/` preserves all six submissions despite their conflicting original filenames. `inputs/creator_ratings.json` preserves publisher-native grades, explicit ordinary grades, conditional alternatives, dated editions and original links. The October joint LLU edition and postlaunch LR rare review are excluded. `inputs/qualitative_sources.json` records the partial Lords of Limited evidence without inventing a whole-set card ranking.

Creator transcription comes from the [community collector](https://chunk.science/mtga-reality-fracture.html), including its preserved September Alex Nikolic and Marc Anderson editions. Dated metadata is evidence rather than cryptographic proof that live publisher pages never changed. The Drosan edition is provisional. LR grades are caption transcriptions, not claims of independently heard audio. `prepare_inputs.py` converts already downloaded scratch sources into the compact inputs; ordinary evaluation needs only the retained inputs and frozen local truth response.

The results measure this early observed GIH table, including drafting, deck selection, and player behavior. They do not estimate causal card strength. Sampling intervals are approximate, the scored card rates can move, and the six original forecast target windows differed. Creator percentage conversion is retrospective calibration, not a claim that creators published percentages before release.
