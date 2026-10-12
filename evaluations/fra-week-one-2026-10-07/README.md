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

`inputs/model_sources.json` identifies each PR, exact commit and original repository path. `inputs/models/` preserves all six submissions despite their conflicting original filenames. `inputs/creator_ratings.json` preserves publisher-native grades, explicit ordinary grades, conditional alternatives, dated editions and original links. The original evaluation excludes the October joint LLU edition and LR release-day rare review. `inputs/qualitative_sources.json` records the partial Lords of Limited evidence without inventing a whole-set card ranking.

Creator transcription comes from the [community collector](https://chunk.science/mtga-reality-fracture.html), including its preserved September Alex Nikolic and Marc Anderson editions. Dated metadata is evidence rather than cryptographic proof that live publisher pages never changed. The Drosan edition is provisional. LR grades are caption transcriptions, not claims of independently heard audio. `prepare_inputs.py` converts already downloaded scratch sources into the compact inputs; ordinary evaluation needs only the retained inputs and frozen local truth response.

The results measure this early observed GIH table, including drafting, deck selection, and player behavior. They do not estimate causal card strength. Sampling intervals are approximate, the scored card rates can move, and the six original forecast target windows differed. Creator percentage conversion is retrospective calibration, not a claim that creators published percentages before release.

## Native creator models, including LR rares

The October 8 update uses `inputs/creator_ratings_native.json`, including 97 captured grades from LR episode 873. This is a user-authorized release-day exception: the official audio page is dated September 29 and the video September 30. LR is one joint LSV + Marshall source. The other nine retained creator editions and the frozen October 7 observed snapshot are unchanged.

```sh
python creator_native.py --truth inputs/local/17lands_fra_all_time.json
node build_creator_workbook.mjs /absolute/output/creator_predictions.xlsx
```

The workbook builder requires `@oai/artifact-tool` in the configured Codex bundled Node dependencies. The statistical fit uses the Python dependencies above. `prepare_inputs.py --include-lr-rares --output inputs/creator_ratings_native.json` can reconstruct the separate rating input from the frozen card and community collector files supplied with its required arguments; it is not needed to run the retained input.

Each creator receives a separate increasing isotonic grade curve with additive context indicators. Numeric grades retain their native values; letters use explicit ordinal modifier steps and split letters use their midpoint. An ordinary grade is selected when present; otherwise the first explicitly graded conditional scenario is used. Bare SB and missing grades remain blank. The collector's semantic grade index never enters these fits.

Bucket counts use distinct cards in each creator's captured rating table, with overlap permitted. A bucket qualifies only when that creator has more than 20 cards in it; eligibility is determined without outcomes and stays fixed across folds. Only LR's 25-card build-around bucket qualifies, with 24 main-set training cards. Its fitted offset is −1.942532 percentage points. The nine other models contain only their native-grade curve. This coefficient describes association conditional on the selected grade; it is not a causal effect of a card being a build-around.

`results/native-creators/` contains:

| File | Purpose |
| --- | --- |
| creator_table.html | All 295 cards, each creator's complete raw grade and final mapped percentage |
| creator_predictions_wide.csv / creator_predictions.xlsx | Complete card table; workbook also contains model curves, counts and sources |
| creator_predictions_long.csv | All 2,950 card/source rows, selected grades, context flags, final mappings, separately held-out mappings, observations and missing reasons |
| creator_models.md / creator_models.html / creator_models.json | Each creator's final model and exact grade knots and context coefficients |
| creator_grade_mappings.csv | Every observed training grade and its fitted baseline |
| context_buckets.csv / context_memberships.csv | Bucket counts and every constituent card |
| creator_scores.csv / validation.json | Held-out scores, snapshot checksums and independent fit verification |

Final table mappings use all eligible main-set cards and are retrospective fitted values. Accuracy uses separate five-fold predictions repeated twenty times, always excluding the scored card. Special Guests and cards with suppressed outcomes receive a mapping if they have a usable grade, but do not train it. The original dashboard retains its initial creator conversion and links to this update explicitly. No numerical model is invented for Lords of Limited or Jim Davis without raw per-card grades.
