"""Reproducible early FRA card evaluation; run with a locally retained 17Lands response.

python evaluate.py --truth inputs/local/17lands_fra_all_time.json
No network calls are made. Missing rates/grades are never imputed for scoring.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
MODEL_LABELS = {
    "gemini-3.8-flash": "Gemini 3.8 Flash", "gpt-6-astra-xhigh": "GPT-6 Astra (xhigh)",
    "claude-fable-5.1": "Claude Fable 5.1", "grok-4.7": "Grok 4.7",
    "claude-opus-5.5": "Claude Opus 5.5", "gpt-6": "GPT-6",
}
SEED, FOLDS, REPEATS, BOOTSTRAPS = 20261007, 5, 20, 2000


def norm(name):
    return name.split(" // ")[0].replace("’", "'").replace("−", "-").strip().casefold()


def load_inputs(truth_path):
    cards = json.loads((ROOT / "inputs/cards.json").read_text())
    catalog = {norm(card["name"]): card for card in cards}
    assert len(catalog) == 295
    frame = pd.DataFrame(cards).set_index("name")
    frame["is_basic"] = frame["type"].str.startswith("Basic Land")
    frame["is_main"] = frame["set"].eq("fra") & ~frame["is_basic"]
    frame["color_label"] = frame["colors"].apply(lambda x: "".join(x) or "C")
    frame["gih_wr"] = np.nan
    frame["gih_n"] = 0
    frame["mtga_id"] = pd.Series(index=frame.index, dtype="object")
    payload = json.loads(truth_path.read_text())
    assert isinstance(payload, dict) and "data" in payload, "Expected the site's /api/card_data response envelope"
    truth_names = set()
    for row in payload["data"]:
        key = norm(row["name"])
        assert key in catalog, f"Unmatched ground-truth card: {row['name']}"
        name = catalog[key]["name"]
        assert name not in truth_names
        truth_names.add(name)
        n, rate = row["ever_drawn_game_count"], row["ever_drawn_win_rate"]
        frame.loc[name, "gih_n"] = n
        frame.loc[name, "mtga_id"] = row["mtga_id"]
        if rate is not None:
            assert 0 <= rate <= 1 and n >= 500
            frame.loc[name, "gih_wr"] = 100 * rate
    assert len(truth_names) == 290
    sources, native = {}, {}
    for info in json.loads((ROOT / "inputs/model_sources.json").read_text()):
        key = info["id"]
        path = ROOT / "inputs" / info["file"]
        if path.suffix == ".csv":
            raw = pd.read_csv(path)
            rows = [(row["name"], row["predicted_gih_wr"]) for _, row in raw.iterrows()]
        else:
            rows = []
            for line in path.read_text().splitlines():
                match = re.match(r"^\|\s*([^|]+?)\s*\|\s*(\d+(?:\.\d+)?%|N/A)\s*\|\s*$", line)
                if match:
                    rows.append((match[1], None if match[2] == "N/A" else float(match[2].rstrip("%"))))
        sources[key] = {"id": key, "name": MODEL_LABELS[key], "kind": "model", "url": f"https://github.com/wanqizhu/mtg-benchmark/pull/{info['pr']}", "commit": info["commit"], "input_rows": len(rows)}
        frame[key] = np.nan
        seen = set()
        native[key] = {}
        for raw_name, value in rows:
            card_key = norm(raw_name)
            assert card_key in catalog, f"Unmatched prediction: {raw_name}"
            name = catalog[card_key]["name"]
            assert name not in seen, f"Duplicate prediction: {name}"
            seen.add(name)
            if value is not None:
                assert 0 <= value <= 100
                frame.loc[name, key] = value
                native[key][name] = f"{value:.1f}%"
        assert frame.loc[frame["is_main"], key].notna().sum() == 280
    creator_data = json.loads((ROOT / "inputs/creator_ratings.json").read_text())
    context_only = {}
    for source in creator_data["sources"]:
        key = source["id"]
        sources[key] = {k: v for k, v in source.items() if k != "rows"}
        frame[key] = np.nan
        native[key], context_only[key] = {}, []
        seen = set()
        for row in source["rows"]:
            card_key = norm(row["name"])
            assert card_key in catalog, f"Unmatched creator card: {row['name']}"
            name = catalog[card_key]["name"]
            assert name not in seen
            seen.add(name)
            native[key][name] = row["native"]
            if row['scope'] in ['ordinary', 'ordinary_plus_conditional']:
                assert row['score'] is not None, f"Explicit ordinary grade discarded: {key} {name}"
            else:
                assert row['score'] is None
            if row["score"] is not None:
                frame.loc[name, key] = row["score"]
            else:
                context_only[key].append(name)
        sources[key]["context_only"] = context_only[key]
        sources[key]["input_rows"] = len(source["rows"])
    model_keys = [key for key, source in sources.items() if source["kind"] == "model"]
    creator_keys = [key for key, source in sources.items() if source["kind"] == "creator"]
    frame["model_ensemble"] = frame[model_keys].mean(axis=1).where(frame[model_keys].notna().all(axis=1))
    frame["human_consensus"] = frame[creator_keys].mean(axis=1).where(frame[creator_keys].notna().sum(axis=1) >= 3)
    sources["model_ensemble"] = {"id": "model_ensemble", "name": "Six-model mean", "kind": "model_ensemble", "url": "", "input_rows": int(frame["model_ensemble"].notna().sum())}
    sources["human_consensus"] = {"id": "human_consensus", "name": "Prerelease creator consensus", "kind": "creator_ensemble", "url": creator_data["collector_url"], "input_rows": int(frame["human_consensus"].notna().sum())}
    native["model_ensemble"], native["human_consensus"] = {}, {}
    return frame, sources, native, creator_data, payload


def fold_assignments(names):
    result = []
    for repeat in range(REPEATS):
        order = sorted(names, key=lambda name: hashlib.sha256(f"{SEED}:{repeat}:{name}".encode()).hexdigest())
        result.append({name: i % FOLDS for i, name in enumerate(order)})
    return result


def calibrate(frame, sources, mask):
    names = frame.index[mask].tolist()
    assignments = fold_assignments(names)
    mappings, folds_ledger = [], []
    for key in sources:
        available = mask & frame[key].notna()
        names_here = frame.index[available].tolist()
        assert len(names_here) >= 50
        totals = pd.Series(0.0, index=names_here)
        appearances = pd.Series(0, index=names_here)
        baseline = pd.Series(0.0, index=names_here)
        for repeat, assignment in enumerate(assignments):
            for fold in range(FOLDS):
                test_names = [name for name in names_here if assignment[name] == fold]
                train_names = [name for name in names_here if assignment[name] != fold]
                assert set(test_names).isdisjoint(train_names)
                assert len(train_names) + len(test_names) == len(names_here)
                model = IsotonicRegression(increasing=True, out_of_bounds="clip")
                model.fit(frame.loc[train_names, key], frame.loc[train_names, "gih_wr"])
                prediction = model.predict(frame.loc[test_names, key])
                totals.loc[test_names] += prediction
                appearances.loc[test_names] += 1
                baseline.loc[test_names] += frame.loc[train_names, "gih_wr"].mean()
                if repeat == 0:
                    folds_ledger.append({"source": key, "fold": fold, "train": len(train_names), "test": len(test_names), "overlap": 0})
        assert appearances.eq(REPEATS).all()
        frame[key + "__cal"] = np.nan
        frame[key + "__baseline"] = np.nan
        frame.loc[names_here, key + "__cal"] = totals / appearances
        frame.loc[names_here, key + "__baseline"] = baseline / appearances
        final = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(frame.loc[available, key], frame.loc[available, "gih_wr"])
        assert np.all(np.diff(final.y_thresholds_) >= -1e-12)
        # This final fitted mapping is descriptive only; primary MAE uses OOF values.
        for value, group in frame.loc[available].groupby(key):
            mappings.append({"source": key, "grade_or_prediction": value, "cards": len(group), "mean_actual_wr": group["gih_wr"].mean(), "monotone_fitted_wr": float(final.predict([value])[0]), "mean_oof_wr": group[key + "__cal"].mean()})
        # Special Guests are never training examples and use the main-set mapping.
        other = ~frame["is_main"] & ~frame["is_basic"] & frame[key].notna()
        if other.any():
            frame.loc[other, key + "__cal"] = final.predict(frame.loc[other, key])
    return mappings, folds_ledger


def stats(x, y, n, calibrated=None):
    x, y, n = np.asarray(x), np.asarray(y), np.asarray(n)
    error = x - y
    out = {"cards": len(x), "pearson_r": float(pearsonr(x, y).statistic), "spearman_rho": float(spearmanr(x, y).statistic), "kendall_tau_b": float(kendalltau(x, y).statistic), "mae_pp": float(np.mean(np.abs(error))), "rmse_pp": float(np.sqrt(np.mean(error**2))), "bias_pp": float(np.mean(error)), "weighted_mae_pp": float(np.average(np.abs(error), weights=n)), "prediction_sd": float(np.std(x, ddof=1)), "actual_sd": float(np.std(y, ddof=1)), "within_2pp": float(np.mean(np.abs(error) <= 2)), "max_abs_error_pp": float(np.max(np.abs(error)))}
    if calibrated is not None:
        residual = np.asarray(calibrated) - y
        out.update({"calibrated_mae_pp": float(np.mean(np.abs(residual))), "calibrated_rmse_pp": float(np.sqrt(np.mean(residual**2))), "calibrated_bias_pp": float(np.mean(residual)), "calibrated_pearson_r": float(pearsonr(calibrated, y).statistic)})
    return out


def bootstrap(x, y, calibrated):
    rng = np.random.default_rng(SEED)
    x, y, calibrated = np.asarray(x), np.asarray(y), np.asarray(calibrated)
    indexes = rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))
    xx, yy = x[indexes], y[indexes]
    xc, yc = xx - xx.mean(axis=1)[:, None], yy - yy.mean(axis=1)[:, None]
    correlations = (xc * yc).sum(axis=1) / np.sqrt((xc**2).sum(axis=1) * (yc**2).sum(axis=1))
    maes = np.abs(xx - yy).mean(axis=1)
    cal_maes = np.abs(calibrated[indexes] - yy).mean(axis=1)
    out = {}
    for key, values in [("pearson", correlations), ("mae", maes), ("cal_mae", cal_maes)]:
        lo, hi = np.quantile(values, [0.025, 0.975])
        out[key + "_ci_low"], out[key + "_ci_high"] = float(lo), float(hi)
    return out


def evaluate_pools(frame, sources):
    observed = frame["gih_wr"].notna() & frame["gih_n"].ge(500)
    primary = observed & frame["is_main"]
    complete_creators = [key for key, source in sources.items() if source["kind"] == "creator" and key != "limitedresources"]
    core = primary & frame[complete_creators].notna().all(axis=1)
    cu = primary & frame["rarity"].isin(["common", "uncommon"])
    cu_core = cu & frame[[*complete_creators, "limitedresources"]].notna().all(axis=1)
    pools = {"main_available": primary, "main_shared_creators": core, "cu_available": cu, "cu_shared_including_lr": cu_core, "main_n1000": primary & frame["gih_n"].ge(1000), "main_n5000": primary & frame["gih_n"].ge(5000), "special_guests": observed & frame["set"].eq("spg")}
    rows = []
    for pool, mask in pools.items():
        for key, source in sources.items():
            if pool == "main_shared_creators" and key == "limitedresources":
                continue  # Partial LR has its own fair matched CU pool.
            selected = mask & frame[key].notna() & frame[key + "__cal"].notna()
            group = frame.loc[selected]
            if len(group) < 3:
                continue
            # Raw human grades have no percentage interpretation: omit raw error metrics.
            numeric = source["kind"].startswith("model")
            score = stats(group[key], group["gih_wr"], group["gih_n"], group[key + "__cal"])
            if not numeric:
                for field in ["mae_pp", "rmse_pp", "bias_pp", "weighted_mae_pp", "within_2pp", "max_abs_error_pp", "prediction_sd"]:
                    score[field] = None
            row = {"pool": pool, "source": key, "name": source["name"], "kind": source["kind"], **score}
            if pool == "main_available":
                bounds = bootstrap(group[key], group["gih_wr"], group[key + "__cal"])
                if not numeric:
                    bounds["mae_ci_low"] = bounds["mae_ci_high"] = None
                row.update(bounds)
                row["mean_only_oof_mae_pp"] = np.mean(np.abs(group[key + "__baseline"] - group["gih_wr"]))
            rows.append(row)
    return pd.DataFrame(rows), pools


def paired_comparisons(frame, sources, pools):
    """Same-card loss differences; negative favors the first predictor."""
    marc = next(k for k, s in sources.items() if s["name"] == "Marc Anderson")
    alex = next(k for k, s in sources.items() if s["name"].startswith("Alex Nikolic"))
    specs = [("main_available", "claude-opus-5.5", k, "raw") for k, s in sources.items() if s["kind"] == "model" and k != "claude-opus-5.5"]
    specs += [("main_available", "claude-opus-5.5", "model_ensemble", "raw")]
    specs += [("main_shared_creators", "claude-opus-5.5", k, "calibrated") for k in [marc, alex, "human_consensus"]]
    specs += [("cu_shared_including_lr", "claude-opus-5.5", "limitedresources", "calibrated")]
    rows = []
    for pool, first, second, scale in specs:
        suffix = "__cal" if scale == "calibrated" else ""
        selected = pools[pool] & frame[first + suffix].notna() & frame[second + suffix].notna()
        group = frame.loc[selected]
        delta = (group[first + suffix] - group["gih_wr"]).abs() - (group[second + suffix] - group["gih_wr"]).abs()
        rng = np.random.default_rng(SEED)
        samples = delta.to_numpy()[rng.integers(0, len(group), size=(BOOTSTRAPS, len(group)))].mean(axis=1)
        lo, hi = np.quantile(samples, [.025, .975])
        rows.append({"pool": pool, "scale": scale, "first": first, "second": second, "first_name": sources[first]["name"], "second_name": sources[second]["name"], "cards": len(group), "mae_difference_pp": delta.mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def group_errors(frame, sources, primary):
    selectors = {}
    for color in frame["color_label"].unique():
        selectors["color:" + color] = frame["color_label"].eq(color)
    for rarity in ["common", "uncommon", "rare", "mythic"]:
        selectors["rarity:" + rarity] = frame["rarity"].eq(rarity)
    selectors.update({
        "mana:0-2": frame["cmc"].le(2) & ~frame["type"].str.contains("Land"),
        "mana:3-4": frame["cmc"].between(3, 4), "mana:5+": frame["cmc"].ge(5),
        "prepared": frame["layout"].eq("prepare"),
        "empower_Jace": frame["oracle"].str.contains("empower", case=False),
        "Way_enchantment": pd.Series(frame.index.str.startswith("Way of the"), index=frame.index),
        "Heartwood_text": frame["oracle"].str.contains("Heartwood", case=False),
        "threshold_text": frame["oracle"].str.contains("threshold", case=False),
        "planeswalker": frame["type"].str.contains("Planeswalker"),
    })
    rows = []
    for key, source in sources.items():
        values = frame[key] if source["kind"].startswith("model") else frame[key + "__cal"]
        for category, selector in selectors.items():
            group = frame.loc[primary & selector & values.notna()]
            if len(group) < 3:
                continue
            error = values.loc[group.index] - group["gih_wr"]
            rows.append({"source": key, "group": category, "cards": len(group), "bias_pp": error.mean(), "mae_pp": error.abs().mean(), "actual_mean_wr": group["gih_wr"].mean()})
    return pd.DataFrame(rows)


def wilson(rate, n):
    p, z = rate / 100, 1.96
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    delta = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return 100*(center-delta), 100*(center+delta)


def mistakes(frame, sources, primary):
    rows = []
    for key, source in sources.items():
        values = frame[key] if source["kind"].startswith("model") else frame[key + "__cal"]
        selected = frame.loc[primary & values.notna()].copy()
        selected["error"] = values.loc[selected.index] - selected["gih_wr"]
        for direction, group in [("overpredicted", selected.nlargest(10, "error")), ("underpredicted", selected.nsmallest(10, "error"))]:
            for name, row in group.iterrows():
                low, high = wilson(row["gih_wr"], row["gih_n"])
                rows.append({"source": key, "direction": direction, "card": name, "prediction_wr": values[name], "actual_wr": row["gih_wr"], "error_pp": row["error"], "gih_n": row["gih_n"], "approx_ci_low": low, "approx_ci_high": high, "rarity": row["rarity"], "colors": row["color_label"]})
    return pd.DataFrame(rows)


def records_for_dashboard(frame, sources, native):
    records = []
    for name, row in frame.iterrows():
        record = {"name": name, "front": name.split(" // ")[0], "rarity": row["rarity"], "color": row["color_label"], "set": row["set"], "cost": row["cost"], "type": row["type"], "oracle": row["oracle"], "url": row["scryfall_uri"], "cmc": row["cmc"], "actual": None if pd.isna(row["gih_wr"]) else row["gih_wr"], "n": int(row["gih_n"]), "values": {}}
        for attr in ["layout", "power", "toughness", "loyalty"]:
            item = row.get(attr)
            record[attr] = None if item is None or pd.isna(item) else item
        if record["actual"] is not None:
            record["interval"] = wilson(record["actual"], record["n"])
        for key, source in sources.items():
            raw, cal = row[key], row[key + "__cal"]
            if pd.isna(raw):
                continue
            record["values"][key] = {"raw": float(raw), "cal": None if pd.isna(cal) else float(cal), "native": native[key].get(name, f"{raw:.2f}")}
        records.append(record)
    return records


def markdown_table(table, columns, digits=3):
    selected = table[columns].copy()
    for column in selected.select_dtypes(include="number").columns:
        selected[column] = selected[column].map(lambda x: "—" if pd.isna(x) else f"{x:.{digits}f}" if isinstance(x, (float, np.floating)) else str(x))
    selected = selected.fillna("—")
    header = "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
    return header + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in selected.values)


def write_report(frame, sources, scores, pools, error_table, group_table, creator_data, manifest, paired):
    main = scores[scores["pool"].eq("main_available")]
    models = main[main["kind"].eq("model")].sort_values("mae_pp")
    creators = main[main["kind"].str.startswith("creator")].sort_values("spearman_rho", ascending=False)
    shared = scores[scores["pool"].eq("main_shared_creators")].sort_values("spearman_rho", ascending=False)
    cu = scores[scores["pool"].eq("cu_shared_including_lr")].sort_values("spearman_rho", ascending=False)
    cols = ["name", "cards", "pearson_r", "spearman_rho", "mae_pp", "bias_pp", "calibrated_mae_pp"]
    text = f"""# Reality Fracture: early-format prediction evaluation

The six immutable September model submissions are evaluated against the 17Lands Premier Draft ALL_TIME table retrieved **{manifest['retrieved_at']}**. This follows the [linked ground-truth page]({manifest['page_url']}). It is the early October 7 snapshot, **not an exact seven-calendar-day window**: the current API uses named time periods and ignores arbitrary `start_date`/`end_date` arguments. No suppressed card rate was reconstructed or invented.

The primary model comparison has **{int(pools['main_available'].sum())} main-set nonbasic cards**, all with reported rates and at least 500 GIH observations. The table reports {int(frame['gih_wr'].notna().sum())}/290 draft-pool card rates. Basic lands have no comparable card-statistic rows. Special Guests are supplemental because only four have a reported rate, and two original models omitted them.

## Six model submissions: original percentage forecasts

{markdown_table(models, cols)}

`mae_pp` is mean absolute error in percentage points, with each card weighted equally. Positive bias means overprediction. Pearson measures linear association; Spearman measures card ordering with average ranks for ties. These are different objectives. All six rows above use the same cards. The six-model mean is a derived aggregate, not a seventh independent submission.

A held-out constant-mean baseline has MAE **{models['mean_only_oof_mae_pp'].iloc[0]:.3f} pp**. Gemini's original percentages have slightly higher MAE than this baseline despite useful positive rank correlation. Fable's original bias is closest to zero among the stronger rank predictors; Opus combines better ordering with an optimistic overall baseline. GPT-6 has the largest positive bias. Calibration improves some sources' absolute errors but can worsen others; it cannot supply missing card discrimination.

Opus has the best observed raw MAE, Pearson correlation, and Spearman correlation among the six models. Fable is second by raw MAE; Astra is second by Spearman. The equal six-model ensemble does not improve on Opus. Paired 95% card-bootstrap intervals below show uncertainty in the MAE differences: negative favors Opus; an interval spanning zero does not establish a clear difference on this set. These are exploratory, unadjusted comparisons; choosing the observed leader and testing several pairs can exaggerate apparent evidence.

{markdown_table(paired[paired['scale'].eq('raw')], ['second_name', 'cards', 'mae_difference_pp', 'ci_low', 'ci_high'])}

At the stricter 5,000-observation threshold, the same six models cover 198 cards and Opus still has the lowest raw MAE. This reduces the influence of unstable sparse-card rates, but changes the card population toward frequently played cards.

{markdown_table(scores[scores['pool'].eq('main_n5000') & scores['kind'].eq('model')].sort_values('mae_pp'), ['name','cards','pearson_r','spearman_rho','mae_pp'])}

## Creator grades converted to win rates

{markdown_table(creators, ['name', 'cards', 'pearson_r', 'spearman_rho', 'calibrated_mae_pp'])}

Correlations here use the **original grade order/index**, not the converted percentages. Coverage differs, so these rows alone are not a fair common-card leaderboard. Limited Resources here is **Marshall Sutcliffe + Luis Scott-Vargas, episode 872 only**, covering prerelease commons/uncommons. Episode 873's rare/mythic grades appeared after Arena launch and are excluded.

For letter grades, Pearson depends on the chosen numeric grade encoding; **Spearman is the primary creator ranking comparison**. The collector's semantic index also harmonizes differently spaced numerical review scales for the derived consensus. Isotonic conversion depends only on order and ties, so changing the spacing of an individual source's grades cannot improve its converted score.

{markdown_table(pd.DataFrame([{'name': s['name'], 'edition_date': s['publication_or_recorded_date'], 'native_scale': s['scale'], 'graded_rows': s['input_rows'], 'context_only': len(s.get('context_only', []))} for s in sources.values() if s['kind']=='creator']), ['name','edition_date','native_scale','graded_rows','context_only'])}

For percentage conversion, an increasing isotonic regression maps each source's ordered grades to mean observed GIH rates. At each grade it begins with the equally weighted mean card rate, pooling adjacent grades when their means contradict the grade ordering. It cannot reverse the published tier order. Five folds are assigned by a fixed seeded hash of card names, shared across sources. Each card is predicted by a mapping trained **without that card**, repeated 20 times, and its held-out predictions are averaged. Missing and context-only grades stay missing. The same conversion is applied to model percentages for a fair comparison after correcting their scale and baseline. No rarity, color, rules text, or individual-card identity enters a conversion.

This is a **retrospective, cross-validated calibration**, not a claim that creators published percentage forecasts. A conversion fitted and scored on the same cards would understate error. `calibration.csv` provides counts, unconstrained grade means, a final descriptive monotone mapping, and held-out means; the final descriptive mapping is never used to score its training cards.

An ordinary grade accompanied by a build-around flag or a separate conditional grade **is retained using its explicit ordinary grade**. For example, `D (build-around A+)` contributes D to the general comparison while preserving the A+ scenario in the source record. Only grades with no general interpretation (sideboard-only or conditional-only) remain unscored. This avoids deleting difficult cards just because the reviewer acknowledged synergy.

## Same cards across broad creator reviews and models

This pool requires an ordinary grade from all nine reviews covering most or all of the main set (excluding partial LR). It has **{int(pools['main_shared_creators'].sum())} cards**. Compare rank correlations directly, or compare the `calibrated_mae_pp` column across models and creators. Raw model MAE retains its original-forecast meaning.

{markdown_table(shared, cols)}

## Commons/uncommons including prerelease LSV + Marshall

This matched pool has **{int(pools['cu_shared_including_lr'].sum())} cards**. Every predictor is evaluated on exactly these cards.

{markdown_table(cu, cols)}

These paired comparisons use identical held-out conversion rules. A near tie in calibrated MAE can coexist with a ranking-correlation difference: coarse grades produce ties, and calibration compresses poorly discriminating scores toward the mean. Each source's main-set mapping is trained on its available main-set cards, even when scoring a smaller matched pool; each scored card is still excluded from every mapping used to predict it.

{markdown_table(paired[paired['scale'].eq('calibrated')], ['pool', 'second_name', 'cards', 'mae_difference_pp', 'ci_low', 'ci_high'])}

## Biggest original model misses

For each model, positive errors are cards forecast too highly; negative errors are cards forecast too low. Full top-ten lists in both directions are in `mistakes.csv`. The examples below require at least 1,000 GIH observations; sparse examples remain in the full file with sample counts.
"""
    robust = frame.loc[pools['main_available'] & frame['gih_n'].ge(1000)].copy()
    robust['error_pp'] = robust['model_ensemble'] - robust['gih_wr']
    selected = pd.concat([robust.nlargest(6, 'error_pp'), robust.nsmallest(6, 'error_pp')]).reset_index().rename(columns={'name':'card','model_ensemble':'mean_prediction_wr','gih_wr':'actual_wr'})
    contextual = "\n## Shared errors and environment context\n\nThe equal six-model mean shows shared failures. The table takes the six largest misses in each direction among cards with at least 1,000 GIH observations.\n\n" + markdown_table(selected, ['card','mean_prediction_wr','actual_wr','error_pp','gih_n'], digits=2) + "\n\n"
    contextual += (ROOT / 'interpretive_notes.md').read_text() + "\n\n"
    important_groups = ['color:U','color:W','color:B','color:R','color:G','color:BU','color:BR','Way_enchantment','empower_Jace','Heartwood_text','planeswalker','mana:5+']
    selected_groups = group_table[group_table['source'].eq('model_ensemble') & group_table['group'].isin(important_groups)]
    contextual += markdown_table(selected_groups, ['group','cards','bias_pp','mae_pp','actual_mean_wr']) + "\n\n"
    text = text.replace('## Biggest original model misses', contextual + '## Biggest original model misses')
    for key in models["source"]:
        robust = error_table[error_table["source"].eq(key) & error_table["gih_n"].ge(1000)]
        selected = pd.concat([robust[robust["direction"].eq("overpredicted")].head(3), robust[robust["direction"].eq("underpredicted")].head(3)])
        text += f"\n### {sources[key]['name']}\n\n" + markdown_table(selected, ["card", "prediction_wr", "actual_wr", "error_pp", "gih_n"], digits=2) + "\n"
    text += '\n## Creator conversion misses\n\nThese are errors after retrospective held-out conversion, not original percentage claims. The same minimum of 1,000 observations applies. Full lists for every creator are in `mistakes.csv`.\n'
    for key in ['human_consensus', 'marcanderson', 'alexnikolic', 'limitedresources']:
        robust = error_table[error_table['source'].eq(key) & error_table['gih_n'].ge(1000)]
        selected = pd.concat([robust[robust['direction'].eq('overpredicted')].head(3), robust[robust['direction'].eq('underpredicted')].head(3)])
        text += f"\n### {sources[key]['name']}\n\n" + markdown_table(selected, ['card','prediction_wr','actual_wr','error_pp','gih_n'], digits=2) + '\n'
    text += """
## Uncertainty, weighting, and interpretation

* `scores.csv` includes 2,000 card-bootstrap intervals for main-pool Pearson correlation and MAE, plus thresholds of 1,000 and 5,000 GIH observations. `paired_comparisons.csv` preserves paired loss-difference intervals. These measure uncertainty from which cards are represented, conditional on the observed table and fixed calibration; they are not player-cluster or future-format confidence intervals. Calibration is not refitted within each bootstrap.
* `weighted_mae_pp` weights by GIH count. It answers a different question from equal-card MAE, favoring common, frequently played cards; it is a sensitivity measure.
* Hover details show a binomial Wilson interval for each reported card rate. Repeated copies, players, and decks are not independent Bernoulli trials, so these intervals are approximate. Early card statistics can move as drafting and deckbuilding improve.
* GIH is conditional on players selecting and drawing a card. Its value mixes intrinsic strength, color/archetype strength, deck selection, and player usage. Group errors and card-rule explanations are descriptive evidence, not a causal experiment.
* The Astra submission specified the first 30 dates and several others described settled all-time data. This is an intentionally early checkpoint, not their final target window. No original forecast was edited after seeing outcomes.
* Context-only/sideboard grades were not forced into an invented normal-deck percentage. `coverage.csv` makes omissions explicit, and matched pools avoid awarding an advantage simply for skipping difficult cards.
* Creator consensus averages ordinary 0–100 ordering indexes with at least three available sources. Sources are correlated (including repeat creators), so this is a useful aggregate rather than independent votes or an uncertainty model.

## Source provenance and reproduction

Models: PRs #8–#13, frozen at their submitted commit hashes in `inputs/model_sources.json`, with each table retained independently so colliding repository filenames do not overwrite predictions. Twenty-one prepared-card aliases are resolved to their front-face name for the 17Lands join; one card is still one row.

Creator grades: [community compilation](https://chunk.science/mtga-reality-fracture.html), retaining each publisher's native grade and source link. The collector snapshot was generated at """ + creator_data["collector_generated_at"] + """. Its preserved September Alex and Marc editions are used, and its October joint LLU edition is excluded. The eight other source editions have prerelease dates recorded in the collector's ledger. The Drosan page is provisional and lacks a publisher archive; see its caveat in `inputs/creator_ratings.json`. The original Andrew Quinn grades were also compared with the locally retained September 24 review extract: every directly matched grade agrees. Podcast grades are collector caption transcriptions with evidence links, not claims of independently heard audio.

Lords of Limited episode 497 is a dated September 21 prerelease guide with selected cards and color rankings, rather than a complete card grade table. The publicly accessible opening explicitly praises Craftwork Crusher. A separate qualitative source record is retained; no fabricated 280-card LoL score or correlation is assigned. Their September 28 episode incorporates early-access games and is excluded from prerelease prediction evidence.

The live 17Lands response and interactive results are retained locally. The response carries a reuse notice; the public game dump was unavailable on October 7. Attribution is to **17Lands**, without implying endorsement. See [usage guidelines](https://www.17lands.com/usage_guidelines) before distributing live card data or the derived interactive table. The analysis branch is local and has not been published.

Run from this directory:

```sh
python evaluate.py --truth inputs/local/17lands_fra_all_time.json
```

Dependencies are in `requirements.txt`. The evaluation makes no network calls. The retained response checksum is recorded in `inputs/ground_truth_manifest.json`; reruns abort if the supplied snapshot differs. `validation.json` records scope, parsing, shared fold counts, and metric checks. `cards_and_predictions.csv` is the complete joined card table, and `predictions_long.csv` is the canonical card/source table with explicit units and native grades; `scores.csv`, `group_errors.csv`, `mistakes.csv`, `coverage.csv`, and `calibration.csv` make every summary inspectable. `dashboard.html` contains the interactive plots and card details.
"""
    (ROOT / "results/report.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "inputs/ground_truth_manifest.json").read_text())
    assert hashlib.sha256(args.truth.read_bytes()).hexdigest() == manifest["sha256"], "Use the frozen snapshot, or explicitly create a new dated evaluation."
    frame, sources, native, creator_data, payload = load_inputs(args.truth)
    primary = frame["is_main"] & frame["gih_wr"].notna() & frame["gih_n"].ge(500)
    mappings, folds_ledger = calibrate(frame, sources, primary)
    scores, pools = evaluate_pools(frame, sources)
    paired = paired_comparisons(frame, sources, pools)
    groups = group_errors(frame, sources, primary)
    errors = mistakes(frame, sources, primary)
    coverage = []
    for key, source in sources.items():
        coverage.append({"source": key, "name": source["name"], "input_rows": source["input_rows"], "numeric_main_predictions": int((frame["is_main"] & frame[key].notna()).sum()), "evaluated_main": int((primary & frame[key].notna()).sum()), "numeric_spg_predictions": int((frame["set"].eq("spg") & frame[key].notna()).sum()), "context_only": len(source.get("context_only", []))})
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    joined = frame.copy()
    long_rows = []
    for key, source in sources.items():
        joined[key + '__native'] = pd.Series(native[key])
        for name, card in frame.iterrows():
            numeric = source['kind'].startswith('model')
            interpreted = card[key] if numeric else card[key + '__cal']
            long_rows.append({'card': name, 'set': card['set'], 'rarity': card['rarity'], 'color': card['color_label'], 'source': key, 'source_name': source['name'], 'source_kind': source['kind'], 'native_prediction_or_grade': native[key].get(name), 'original_value': card[key], 'original_unit': 'win_rate_percent' if numeric else 'grade_index_0_100', 'predicted_wr': interpreted, 'calibrated_wr': card[key + '__cal'], 'observed_gih_wr': card['gih_wr'], 'gih_n': int(card['gih_n']), 'error_pp': interpreted-card['gih_wr'], 'scored_main': bool(primary[name] and pd.notna(interpreted)), 'source_url': source['url']})
    joined.to_csv(output / "cards_and_predictions.csv")
    pd.DataFrame(long_rows).to_csv(output / "predictions_long.csv", index=False)
    scores.to_csv(output / "scores.csv", index=False)
    groups.to_csv(output / "group_errors.csv", index=False)
    errors.to_csv(output / "mistakes.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(mappings).to_csv(output / "calibration.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    model_keys = [key for key, source in sources.items() if source["kind"] == "model"]
    # Independently check reported metrics against a direct arithmetic expression.
    for key in model_keys:
        row = scores[scores["pool"].eq("main_available") & scores["source"].eq(key)].iloc[0]
        assert abs(row["mae_pp"] - sum(abs(frame.loc[name, key] - frame.loc[name, "gih_wr"]) for name in frame.index[primary]) / primary.sum()) < 1e-10
    for pool in ['main_shared_creators', 'cu_shared_including_lr']:
        assert scores.loc[scores['pool'].eq(pool), 'cards'].eq(pools[pool].sum()).all()
    uncommons = frame.loc[primary & frame['rarity'].eq('uncommon')]
    assert uncommons['gih_wr'].idxmax() == 'Craftwork Crusher'
    assert (frame.loc['Ajani Unrelenting', model_keys] < frame.loc['Ajani Unrelenting', 'gih_wr']).all()
    validation = {"snapshot_sha256": manifest["sha256"], "catalog_unique": len(frame), "main_nonbasic": int(frame["is_main"].sum()), "draft_pool_rows": len(payload["data"]), "observed_rates": int(frame["gih_wr"].notna().sum()), "primary_cards": int(primary.sum()), "all_six_models_cover_primary": bool(frame.loc[primary, model_keys].notna().all().all()), "pool_sizes": {key: int(mask.sum()) for key, mask in pools.items()}, "input_checksums": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'inputs').rglob('*')) if p.is_file() and 'local' not in p.parts}, "qualitative_check": {"craftwork_crusher_uncommon_gih_rank": 1, "reported_uncommons": len(uncommons), "ajani_underpredicted_by_all_six": True}, "calibration": {"seed": SEED, "folds": FOLDS, "repeats": REPEATS, "training_test_overlap": 0, "held_out_predictions_per_card": REPEATS, "fold_ledger_first_repeat": folds_ledger}, "checks": ["No unmatched or duplicate names", "No suppressed GIH rate imputed", "Model percentage scale 0–100 checked", "All main-set model predictions present", "Explicit ordinary grades retained including conditional flags", "Context-only grades left unscored", "Shared leaderboard card counts agree", "Final mapping monotonicity checked", "Every scored calibration prediction excludes that card", "Raw model MAE independently recomputed", "Postlaunch creator editions excluded"]}
    (output / "validation.json").write_text(json.dumps(validation, indent=2))
    data = {"sources": sources, "cards": records_for_dashboard(frame, sources, native), "scores": json.loads(scores.to_json(orient="records")), "groups": json.loads(groups.to_json(orient="records")), "manifest": manifest, "pool_sizes": validation["pool_sizes"]}
    (output / "dashboard_data.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
    write_report(frame, sources, scores, pools, errors, groups, creator_data, manifest, paired)
    from render_dashboard import render
    render(data, output / "dashboard.html")
    print(scores[scores["pool"].eq("main_available")][["name", "cards", "pearson_r", "spearman_rho", "mae_pp", "calibrated_mae_pp"]].to_string(index=False))
    print("Validation:", validation["pool_sizes"])


if __name__ == "__main__":
    main()
