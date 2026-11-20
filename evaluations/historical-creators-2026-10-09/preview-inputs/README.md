# Draft prediction comparison

`trk-frozen.json` retains the eight prediction tables by commit, the
311-card main-set catalog, and the matched native creator grades. The creator
entries record the source editions, links, and evidence for the revision check.
Casual Spike remains unscored until a Star Trek list can be attributed to the
same reviewer. The build replaces the initial Alex-only grades with the Alex +
Marc joint edition in `alex-joint-2026-11-08.json`, selected at November 8,
2026 22:18:32 UTC, before the November 10 Arena launch. All 336 grades match
the change history through that timestamp; its 306 main-set grades use Alex's
existing historical mapping. Later revisions are excluded.

On November 15, I rechecked Casual Spike’s public sources. Their [November 13
Reddit comment](https://www.reddit.com/r/lrcast/comments/1xvghov/site_to_make_personal_card_rankingstierlist/19sl5uwdcky9v/)
links a [grading template](https://docs.google.com/spreadsheets/d/1Dd30XdrKnFVdYLUgLAwF0FQCeRlOHj3SzdQLH4DBwWw/edit)
with no card names or predictions filled in. Their [Prediction Accuracy tracker](https://docs.google.com/spreadsheets/d/1FxbP7N9wBxtm0fNTkB0jNmAcxx_cbwmH2KEZAR_wQUE/edit)
ends at Reality Fracture, and the November 5 archived and current Star Trek
review indexes do not attribute grades to Casual Spike. The [MTG Rebellion
Discord](https://discord.gg/MkVuMc7Ddm) linked from the FRA review requires
sign-in; its channel history was not checked.

On November 19, I refreshed both sets from the cumulative 17Lands Premier
Draft API at 18:28:59 UTC (10:28:59 AM PST). Each source now scores all its own
rated main-set cards with usable outcomes: 276–282 cards for Reality Fracture
and 295 for every Star Trek source. The six initial Star Trek model submissions
and all creator calibrations are retained.

On November 20, [Muse Spark](https://github.com/wanqizhu/mtg-win-rate-eval/pull/11)
and [Gemini 4 Argon](https://github.com/wanqizhu/mtg-win-rate-eval/pull/12)
were added and scored against the November 19 outcome snapshot. Both PRs
were submitted after release. The site pairs the PR labeled Muse Spark with
the existing Muse Spark 1.3 row. Argon's 30 Stardates rows are outside the
main-set scoring pool.

The site's existing Reality Fracture export supplies the exact creator model
coefficients. The build never fits a model. The single local page uses reported
17Lands Premier Draft GIH with at least 500 reported observations per card.
Each source is scored independently on every card it rated with an eligible
outcome. Basics and Stardates are excluded. The table shows only Pearson r,
with FRA and Star Trek columns. Selecting a score updates the original card
plot and card table. `trk-sandbox-prompt.txt` is the exact prompt supplied by the
user from the Codex Cloud run on November 19. The site displays one shared
sandbox prompt with the differences in brackets.

To rescore, save a new Star Trek response from
`https://www.17lands.com/api/card_data?expansion=TRK&event_type=PremierDraft&time_period=ALL_TIME`
and run from the benchmark checkout, substituting its path for `NEW_TRK_RESPONSE`:

```sh
python3 evaluations/historical-creators-2026-10-09/star_trek_preview.py build \
  --frozen evaluations/historical-creators-2026-10-09/preview-inputs/trk-frozen.json \
  --fra-original ../mtg-benchmark-site/data/draft-predictions.json \
  --trk-raw NEW_TRK_RESPONSE \
  --site-dir ../mtg-benchmark-site
```

Add `--fra-raw NEW_FRA_RESPONSE` to refresh the local Reality Fracture results
as well. Add `--public-summary` to build a Star Trek export with the predictions
and aggregate scores, omitting the observed rates and game counts for each card.
The full local Star Trek export takes precedence when both are present. Preserve file
modification times if copying existing responses: the export records the time
the snapshot was saved, not the most recent game date. The inputs and outputs
containing early card-level outcome data are local and ignored by Git. See
[17Lands usage guidance](https://www.17lands.com/usage_guidelines) before
publishing a version of the card-level comparison.
