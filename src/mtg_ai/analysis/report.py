from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import re
from pathlib import Path

from mtg_ai.analysis.charts import ascii_line_chart, bar_chart
from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.game.session import GameSession
from mtg_ai.eval.tournament import Entrant, round_robin
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy
from mtg_ai.training.obs import encode_action
from mtg_ai.training.value_iteration_baseline import evaluate_solver_alignment


def _pseudo_elo_from_winrate(winrate: float) -> float:
    bounded = min(0.999, max(0.001, winrate))
    return 1200.0 + 400.0 * math.log10(bounded / (1.0 - bounded))


def read_training_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_checkpoint(training_dir: Path) -> Path | None:
    checkpoints = sorted(training_dir.glob("checkpoint_ep*.json"))
    weighted_checkpoints: list[Path] = []
    for checkpoint in checkpoints:
        with checkpoint.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "weights" in payload:
            weighted_checkpoints.append(checkpoint)
    checkpoints = weighted_checkpoints
    if not checkpoints:
        candidate = training_dir / "policy_final.json"
        if candidate.exists():
            return candidate
        return None
    return checkpoints[-1]


def weighted_checkpoints(training_dir: Path) -> list[Path]:
    checkpoints = sorted(training_dir.glob("checkpoint_ep*.json"))
    result: list[Path] = []
    for checkpoint in checkpoints:
        with checkpoint.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "weights" in payload:
            result.append(checkpoint)
    return result


def generate_report(
    training_dir: Path,
    deck_search_csv: Path,
    output_path: Path,
    seed: int,
    generalization_csv: Path | None = None,
    sweep_csv: Path | None = None,
    solver_tune_history_csv: Path | None = None,
    deck_vs_starter_csv: Path | None = None,
    policy_selection_csv: Path | None = None,
    policy_matrix_csv: Path | None = None,
    milestones_csv: Path | None = None,
    convergence_csv: Path | None = None,
    champion_json: Path | None = None,
    champion_csv: Path | None = None,
    significance_csv: Path | None = None,
    replay_index_csv: Path | None = None,
    manifest_json: Path | None = None,
    optimality_audit_json: Path | None = None,
    healthcheck_csv: Path | None = None,
) -> None:
    metrics = read_training_metrics(training_dir / "training_metrics.csv")
    checkpoint = latest_checkpoint(training_dir)

    lines: list[str] = []
    lines.append("# MTG AI Report")
    lines.append("")
    lines.append("## Training progression")
    if metrics:
        points_random: list[tuple[int, float]] = []
        points_mixed: list[tuple[int, float]] = []
        fitness_points: list[tuple[int, float]] = []
        for row in metrics:
            episode = int(row.get("episode_generation", row.get("episode", "0")))
            if row["winrate_vs_random"]:
                points_random.append((episode, float(row["winrate_vs_random"])))
            if row["winrate_vs_mixed"]:
                points_mixed.append((episode, float(row["winrate_vs_mixed"])))
            fitness_value = row.get("fitness")
            if fitness_value:
                fitness_points.append((episode, float(fitness_value)))
        lines.append("### Win rate vs random over training")
        lines.append("```")
        lines.append(ascii_line_chart(points_random))
        lines.append("```")
        lines.append("### Win rate vs mixed baseline over training")
        lines.append("```")
        lines.append(ascii_line_chart(points_mixed))
        lines.append("```")
        if points_mixed:
            lines.append(f"Final pseudo-Elo vs mixed baseline: {_pseudo_elo_from_winrate(points_mixed[-1][1]):.2f}")
        if len(fitness_points) >= 10:
            trailing = fitness_points[-10:]
            leading = fitness_points[-20:-10] or trailing
            leading_avg = sum(value for _, value in leading) / len(leading)
            trailing_avg = sum(value for _, value in trailing) / len(trailing)
            delta = trailing_avg - leading_avg
            lines.append(
                "Convergence check (fitness delta between recent windows): "
                f"{delta:+.4f}"
            )
    else:
        lines.append("No training metrics found.")

    lines.append("")
    lines.append("## League Elo snapshot")
    if checkpoint is not None:
        policy = WeightedHeuristicPolicy.load(checkpoint)
        trained = Entrant("trained", policy, ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9)
        entrants = [
            Entrant("random", RandomBot(), trained.deck),
            Entrant("bolt_face", BoltFaceBot(), trained.deck),
            Entrant("goblin_aggro", GoblinAggroBot(), trained.deck),
            Entrant("mixed_bolt_face", MixedBoltFaceBot(), trained.deck),
            trained,
        ]
        _, ratings = round_robin(
            entrants=entrants,
            games_per_pair=60,
            config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=seed),
            seed=seed,
        )
        bars = sorted(ratings.items(), key=lambda item: item[1], reverse=True)
        lines.append("```")
        lines.append(bar_chart(bars))
        lines.append("```")
    else:
        lines.append("No checkpoint found.")

    lines.append("")
    lines.append("## Checkpoint league progression")
    checkpoints = weighted_checkpoints(training_dir)
    if len(checkpoints) >= 2:
        checkpoint_entrants: list[Entrant] = []
        for checkpoint in checkpoints:
            policy = WeightedHeuristicPolicy.load(checkpoint)
            checkpoint_entrants.append(
                Entrant(
                    checkpoint.stem,
                    policy,
                    ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9,
                )
            )
        _, checkpoint_ratings = round_robin(
            entrants=checkpoint_entrants,
            games_per_pair=24,
            config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=seed + 500),
            seed=seed + 500,
        )
        points: list[tuple[int, float]] = []
        for checkpoint in checkpoints:
            match = re.search(r"ep(\d+)", checkpoint.stem)
            if match is None:
                continue
            episode = int(match.group(1))
            points.append((episode, checkpoint_ratings[checkpoint.stem]))
        points.sort(key=lambda item: item[0])
        lines.append("```")
        lines.append(ascii_line_chart(points))
        lines.append("```")
    else:
        lines.append("Not enough checkpoints for progression chart.")

    lines.append("")
    lines.append("## Deck search summary")
    if deck_search_csv.exists():
        with deck_search_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            lines.append(
                f"- life={row['life']} hand={row['opening_hand']}: "
                f"M={row['mountains']} B={row['bolts']} G={row['goblins']} score={row['score']}"
            )
    else:
        lines.append("No deck search CSV found.")

    lines.append("")
    lines.append("## Best deck vs starter validation")
    if deck_vs_starter_csv is not None and deck_vs_starter_csv.exists():
        with deck_vs_starter_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            lines.append(
                f"- life={row['life']} hand={row['opening_hand']} deck={row['best_deck']} "
                f"win_rate={row['best_vs_starter_win_rate']} ({row['wins']}-{row['losses']}-{row['draws']})"
            )
    else:
        lines.append("No best-deck-vs-starter CSV found.")

    lines.append("")
    lines.append("## Generalization matrix (trained vs baselines)")
    if generalization_csv is not None and generalization_csv.exists():
        with generalization_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            lines.append(
                f"- life={row['life']} hand={row['opening_hand']} vs {row['opponent']}: "
                f"win_rate={row['trained_win_rate']} ({row['wins']}-{row['losses']}-{row['draws']})"
            )
    else:
        lines.append("No generalization matrix CSV found.")

    lines.append("")
    lines.append("## Decision differences on sampled states")
    if checkpoint is not None:
        trained_policy = WeightedHeuristicPolicy.load(checkpoint)
        sample_session = GameSession(
            decks={
                0: ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9,
                1: ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9,
            },
            controllers={0: RandomBot(), 1: RandomBot()},
            config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=seed),
            seed=seed,
        )
        sampled_states: list[tuple] = []
        for _ in range(240):
            if sample_session.state.game_over or len(sampled_states) >= 4:
                break
            actor = sample_session.state.turn.priority_player_id
            legal = sample_session.engine.legal_actions(sample_session.state, actor)
            if actor == 0 and len(legal) > 1:
                sampled_states.append((copy.deepcopy(sample_session.state), legal))
            sample_session.step()
        bots = {
            "trained": trained_policy,
            "mixed": MixedBoltFaceBot(),
            "bolt": BoltFaceBot(),
            "goblin": GoblinAggroBot(),
        }
        for idx, (state, legal_actions) in enumerate(sampled_states, start=1):
            lines.append(f"- State {idx}: step={state.turn.step.value} P0 life={state.players[0].life} P1 life={state.players[1].life}")
            for name, bot in bots.items():
                action = bot.choose_action(state, legal_actions, __import__('random').Random(seed + idx))
                lines.append(f"  - {name}: {encode_action(action, 0)}")
    else:
        lines.append("No checkpoint available for decision comparison.")

    lines.append("")
    lines.append("## Solver alignment in tractable low-life regime")
    if checkpoint is not None:
        trained_policy = WeightedHeuristicPolicy.load(checkpoint)
        alignment = evaluate_solver_alignment(
            trained_policy=trained_policy,
            seed=seed,
            life=10,
            opening_hand=4,
            depth=7,
            sample_count=12,
        )
        lines.append(
            f"- Compared states: {alignment.compared_states}\n"
            f"- Trained agreement: {alignment.trained_agreement:.3f}\n"
            f"- Mixed baseline agreement: {alignment.mixed_agreement:.3f}\n"
            f"- Bolt baseline agreement: {alignment.bolt_agreement:.3f}\n"
            f"- Goblin baseline agreement: {alignment.goblin_agreement:.3f}"
        )
    else:
        lines.append("No checkpoint available for solver alignment.")

    lines.append("")
    lines.append("## Hyperparameter sweep summary")
    if sweep_csv is not None and sweep_csv.exists():
        with sweep_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            top_rows = rows[: min(5, len(rows))]
            for row in top_rows:
                lines.append(
                    f"- run {row['run_id']} seed={row['seed']} eps={row['epsilon']} "
                    f"solver_w={row['solver_weight']} history_w={row['history_weight']} "
                    f"obj={row['objective']} base={row['baseline_avg_winrate']} "
                    f"mixed_floor={row['mixed_floor_winrate']} align={row['solver_alignment']}"
                )
        else:
            lines.append("Sweep CSV is empty.")
    else:
        lines.append("No hyperparameter sweep CSV found.")

    lines.append("")
    lines.append("## Solver-tuning trajectory")
    if solver_tune_history_csv is not None and solver_tune_history_csv.exists():
        with solver_tune_history_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            points = [(int(row["iteration"]), float(row["objective"])) for row in rows]
            lines.append("```")
            lines.append(ascii_line_chart(points))
            lines.append("```")
            final = rows[-1]
            lines.append(
                f"Final tuned objective={final['objective']} "
                f"baseline_avg={final['baseline_avg']} mixed_floor={final['mixed_floor']} "
                f"solver_alignment={final['solver_alignment']}"
            )
        else:
            lines.append("Solver tuning history CSV is empty.")
    else:
        lines.append("No solver tuning history CSV found.")

    lines.append("")
    lines.append("## Policy selection frontier")
    if policy_selection_csv is not None and policy_selection_csv.exists():
        with policy_selection_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            for row in rows[: min(6, len(rows))]:
                lines.append(
                    f"- {row['name']}: objective={row['objective']} baseline_avg={row['baseline_avg']} "
                    f"mixed_floor={row['mixed_floor']} solver_alignment={row['solver_alignment']}"
                )
        else:
            lines.append("Policy selection CSV is empty.")
    else:
        lines.append("No policy selection CSV found.")

    lines.append("")
    lines.append("## Policy selection top-league Elo")
    if policy_matrix_csv is not None and policy_matrix_csv.exists():
        with policy_matrix_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        elo_rows = [row for row in rows if row["b"] == "__elo__"]
        if elo_rows:
            elo_items = sorted(((row["a"], float(row["win_rate_a"])) for row in elo_rows), key=lambda item: item[1], reverse=True)
            lines.append("```")
            lines.append(bar_chart(elo_items))
            lines.append("```")
        else:
            lines.append("No Elo rows found in policy matrix CSV.")
    else:
        lines.append("No policy matrix CSV found.")

    lines.append("")
    lines.append("## Milestone checklist")
    if milestones_csv is not None and milestones_csv.exists():
        with milestones_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        passed = sum(int(row["passed"]) for row in rows)
        total = len(rows)
        lines.append(f"- Passed {passed}/{total} milestones")
        for row in rows:
            status = "PASS" if int(row["passed"]) else "FAIL"
            lines.append(
                f"  - [{status}] {row['milestone']}: metric={row['metric']} threshold={row['threshold']} ({row['details']})"
            )
    else:
        lines.append("No milestones CSV found.")

    lines.append("")
    lines.append("## Checkpoint convergence analysis")
    if convergence_csv is not None and convergence_csv.exists():
        with convergence_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            deltas = [float(row["elo_delta_to"]) for row in rows]
            trailing = deltas[-3:] if len(deltas) >= 3 else deltas
            trailing_avg = sum(trailing) / len(trailing)
            lines.append(
                f"- Mean Elo delta for latest checkpoints: {trailing_avg:+.2f} "
                f"(positive means later checkpoints still improving)"
            )
            points = [(idx + 1, delta) for idx, delta in enumerate(deltas)]
            lines.append("```")
            lines.append(ascii_line_chart(points))
            lines.append("```")
            for row in rows[-3:]:
                lines.append(
                    f"- {row['from']} -> {row['to']}: win_rate_from={row['win_rate_from']} "
                    f"elo_delta_to={row['elo_delta_to']}"
                )
        else:
            lines.append("Convergence CSV is empty.")
    else:
        lines.append("No convergence CSV found.")

    lines.append("")
    lines.append("## Recommended champion policy")
    if champion_json is not None and champion_json.exists():
        payload = json.loads(champion_json.read_text(encoding="utf-8"))
        lines.append(
            f"- {payload['name']} objective={payload['robust_objective']} "
            f"baseline_avg={payload['baseline_avg']} mixed_floor={payload['mixed_floor']} "
            f"solver_alignment={payload['solver_alignment']}"
        )
    else:
        lines.append("No champion JSON found.")

    lines.append("")
    lines.append("## Champion candidate leaderboard")
    if champion_csv is not None and champion_csv.exists():
        with champion_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows[: min(5, len(rows))]:
            lines.append(
                f"- {row['name']}: robust={row['robust_objective']} baseline={row['baseline_avg']} "
                f"mixed_floor={row['mixed_floor']} align={row['solver_alignment']}"
            )
    else:
        lines.append("No champion leaderboard CSV found.")

    lines.append("")
    lines.append("## Head-to-head significance among top candidates")
    if significance_csv is not None and significance_csv.exists():
        with significance_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            significant = [row for row in rows if int(row["significant_vs_0_5"]) == 1]
            lines.append(f"- Significant pairwise outcomes: {len(significant)}/{len(rows)}")
            for row in rows[: min(6, len(rows))]:
                lines.append(
                    f"- {row['a']} vs {row['b']}: wr_a={row['win_rate_a']} "
                    f"CI=[{row['ci95_low']}, {row['ci95_high']}] favored={row['favored']} "
                    f"significant={row['significant_vs_0_5']}"
                )
        else:
            lines.append("Significance CSV is empty.")
    else:
        lines.append("No significance CSV found.")

    lines.append("")
    lines.append("## Replay sample index")
    if replay_index_csv is not None and replay_index_csv.exists():
        with replay_index_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            lines.append(
                f"- {row['matchup']}: winner={row['winner']} result={row['result']} "
                f"turns={row['turns']} actions={row['actions']} replay={row['replay_path']}"
            )
    else:
        lines.append("No replay index CSV found.")

    lines.append("")
    lines.append("## Artifact manifest")
    if manifest_json is not None and manifest_json.exists():
        payload = json.loads(manifest_json.read_text(encoding="utf-8"))
        lines.append(f"- git_head: {payload.get('git_head', 'unknown')}")
        lines.append(f"- created_at_utc: {payload.get('created_at_utc', 'unknown')}")
        records = payload.get("records", [])
        present = [record for record in records if record.get("exists")]
        lines.append(f"- recorded artifacts: {len(records)} (present: {len(present)})")
        for record in present[: min(8, len(present))]:
            lines.append(
                f"  - {record['path']} bytes={record['bytes']} sha256={record['sha256'][:16]}..."
            )
    else:
        lines.append("No artifact manifest JSON found.")

    lines.append("")
    lines.append("## Optimality audit")
    if optimality_audit_json is not None and optimality_audit_json.exists():
        payload = json.loads(optimality_audit_json.read_text(encoding="utf-8"))
        lines.append(f"- champion: {payload['champion_name']}")
        lines.append(
            f"- solver_alignment={payload['solver_alignment']:.4f} "
            f"best_baseline_alignment={payload['best_baseline_alignment']:.4f} "
            f"solver_gap={payload['solver_gap_to_best_baseline']:.4f}"
        )
        lines.append(
            f"- convergence_trailing_mean_elo_delta={payload['convergence_trailing_mean_elo_delta']:+.4f} "
            f"objective_margin={payload['champion_objective_margin']:.4f}"
        )
        lines.append(
            f"- checks: solver_gap={payload['pass_solver_gap']} "
            f"convergence={payload['pass_convergence']} margin={payload['pass_margin']}"
        )
        lines.append(f"- confidence={payload['confidence']:.4f}")
    else:
        lines.append("No optimality audit JSON found.")

    lines.append("")
    lines.append("## Healthcheck summary")
    if healthcheck_csv is not None and healthcheck_csv.exists():
        with healthcheck_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        passed = sum(int(row["passed"]) for row in rows)
        lines.append(f"- Passed checks: {passed}/{len(rows)}")
        for row in rows:
            status = "PASS" if int(row["passed"]) else "FAIL"
            lines.append(f"  - [{status}] {row['name']}: {row['detail']}")
    else:
        lines.append("No healthcheck CSV found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aggregate MTG AI report.")
    parser.add_argument("--training-dir", default="artifacts/training/self_play")
    parser.add_argument("--deck-search-csv", default="artifacts/deck_search/results.csv")
    parser.add_argument("--generalization-csv", default="artifacts/eval/generalization_matrix.csv")
    parser.add_argument("--sweep-csv")
    parser.add_argument("--solver-tune-history-csv")
    parser.add_argument("--deck-vs-starter-csv")
    parser.add_argument("--policy-selection-csv")
    parser.add_argument("--policy-matrix-csv")
    parser.add_argument("--milestones-csv")
    parser.add_argument("--convergence-csv")
    parser.add_argument("--champion-json")
    parser.add_argument("--champion-csv")
    parser.add_argument("--significance-csv")
    parser.add_argument("--replay-index-csv")
    parser.add_argument("--manifest-json")
    parser.add_argument("--optimality-audit-json")
    parser.add_argument("--healthcheck-csv")
    parser.add_argument("--output", default="artifacts/report.md")
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()
    generate_report(
        training_dir=Path(args.training_dir),
        deck_search_csv=Path(args.deck_search_csv),
        generalization_csv=Path(args.generalization_csv),
        sweep_csv=Path(args.sweep_csv) if args.sweep_csv else None,
        solver_tune_history_csv=Path(args.solver_tune_history_csv) if args.solver_tune_history_csv else None,
        deck_vs_starter_csv=Path(args.deck_vs_starter_csv) if args.deck_vs_starter_csv else None,
        policy_selection_csv=Path(args.policy_selection_csv) if args.policy_selection_csv else None,
        policy_matrix_csv=Path(args.policy_matrix_csv) if args.policy_matrix_csv else None,
        milestones_csv=Path(args.milestones_csv) if args.milestones_csv else None,
        convergence_csv=Path(args.convergence_csv) if args.convergence_csv else None,
        champion_json=Path(args.champion_json) if args.champion_json else None,
        champion_csv=Path(args.champion_csv) if args.champion_csv else None,
        significance_csv=Path(args.significance_csv) if args.significance_csv else None,
        replay_index_csv=Path(args.replay_index_csv) if args.replay_index_csv else None,
        manifest_json=Path(args.manifest_json) if args.manifest_json else None,
        optimality_audit_json=Path(args.optimality_audit_json) if args.optimality_audit_json else None,
        healthcheck_csv=Path(args.healthcheck_csv) if args.healthcheck_csv else None,
        output_path=Path(args.output),
        seed=args.seed,
    )
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()

