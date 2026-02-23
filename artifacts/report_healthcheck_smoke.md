# MTG AI Report

## Training progression
### Win rate vs random over training
```
              * * * * * ** * * * * * ** * * * * * ** * * * * * *
                                                                
                                                                
                                                                
                                                                
         * *                                                    
                                                                
   * * *                                                        
 *                                                              
            *                                                   
                                                                
*                                                               
x:[1, 35] y:[0.750, 0.938]
```
### Win rate vs mixed baseline over training
```
       *                                                        
              * * * * * ** * * * * * ** * * * * * ** * * * * * *
                                                                
            *                                                   
                                                                
   * *                                                          
         * *                                                    
                                                                
                                                                
                                                                
                                                                
**                                                              
x:[1, 35] y:[0.542, 0.667]
```
Final pseudo-Elo vs mixed baseline: 1312.29
Convergence check (fitness delta between recent windows): +0.0000

## League Elo snapshot
```
trained                  ################################################ 1229.756
mixed_bolt_face          ############################################### 1222.385
goblin_aggro             ############################################## 1187.047
random                   ############################################## 1186.034
bolt_face                ############################################# 1174.777
```

## Checkpoint league progression
```
                               *                                
                                                                
                                                                
                                                                
                                                                
                                                                
                     *                                          
                                                                
*                                                              *
                                                                
                                                                
          *                               *         *           
x:[5, 35] y:[1196.575, 1207.194]
```

## Deck search summary
- life=5 hand=5: M=5 B=3 G=4 score=0.9062
- life=5 hand=7: M=4 B=2 G=6 score=0.9062
- life=10 hand=5: M=3 B=6 G=3 score=0.9688
- life=10 hand=7: M=4 B=7 G=1 score=0.9375
- life=20 hand=5: M=3 B=7 G=2 score=0.9531
- life=20 hand=7: M=3 B=6 G=3 score=1.0000

## Best deck vs starter validation
- life=5 hand=5 deck=M5/B3/G4 win_rate=0.6000 (60-40-0)
- life=5 hand=7 deck=M4/B2/G6 win_rate=0.5700 (57-43-0)
- life=10 hand=5 deck=M3/B6/G3 win_rate=0.5300 (53-47-0)
- life=10 hand=7 deck=M4/B7/G1 win_rate=0.5400 (54-46-0)
- life=20 hand=5 deck=M3/B7/G2 win_rate=0.7150 (71-28-1)
- life=20 hand=7 deck=M3/B6/G3 win_rate=0.7800 (78-22-0)

## Generalization matrix (trained vs baselines)
- life=5 hand=5 vs random: win_rate=0.7667 (92-28-0)
- life=5 hand=5 vs bolt: win_rate=0.6000 (72-48-0)
- life=5 hand=5 vs goblin: win_rate=0.9250 (111-9-0)
- life=5 hand=5 vs mixed: win_rate=0.4750 (57-63-0)
- life=5 hand=7 vs random: win_rate=0.7583 (91-29-0)
- life=5 hand=7 vs bolt: win_rate=0.5667 (68-52-0)
- life=5 hand=7 vs goblin: win_rate=0.9833 (118-2-0)
- life=5 hand=7 vs mixed: win_rate=0.5000 (60-60-0)
- life=10 hand=5 vs random: win_rate=0.8708 (104-15-1)
- life=10 hand=5 vs bolt: win_rate=0.8958 (107-12-1)
- life=10 hand=5 vs goblin: win_rate=0.9500 (114-6-0)
- life=10 hand=5 vs mixed: win_rate=0.5917 (71-49-0)
- life=10 hand=7 vs random: win_rate=0.9500 (114-6-0)
- life=10 hand=7 vs bolt: win_rate=0.8583 (103-17-0)
- life=10 hand=7 vs goblin: win_rate=0.9750 (117-3-0)
- life=10 hand=7 vs mixed: win_rate=0.5417 (65-55-0)
- life=20 hand=5 vs random: win_rate=0.6708 (41-0-79)
- life=20 hand=5 vs bolt: win_rate=0.8542 (85-0-35)
- life=20 hand=5 vs goblin: win_rate=0.8083 (83-9-28)
- life=20 hand=5 vs mixed: win_rate=0.5833 (58-38-24)
- life=20 hand=7 vs random: win_rate=0.7667 (64-0-56)
- life=20 hand=7 vs bolt: win_rate=0.9625 (111-0-9)
- life=20 hand=7 vs goblin: win_rate=0.9333 (110-6-4)
- life=20 hand=7 vs mixed: win_rate=0.6167 (72-44-4)

## Decision differences on sampled states
- State 1: step=precombat_main P0 life=7 P1 life=10
  - trained: play_land
  - mixed: play_land
  - bolt: play_land
  - goblin: play_land
- State 2: step=precombat_main P0 life=7 P1 life=10
  - trained: cast_goblin
  - mixed: cast_bolt_face
  - bolt: cast_bolt_face
  - goblin: cast_goblin
- State 3: step=begin_combat P0 life=7 P1 life=10
  - trained: cast_bolt_face
  - mixed: cast_bolt_face
  - bolt: cast_bolt_face
  - goblin: pass_priority
- State 4: step=combat_damage P0 life=7 P1 life=10
  - trained: cast_bolt_face
  - mixed: cast_bolt_face
  - bolt: cast_bolt_face
  - goblin: pass_priority

## Solver alignment in tractable low-life regime
- Compared states: 12
- Trained agreement: 0.833
- Mixed baseline agreement: 1.000
- Bolt baseline agreement: 1.000
- Goblin baseline agreement: 0.833

## Hyperparameter sweep summary
- run 1 seed=79 eps=0.22 solver_w=0.0 history_w=0.0 obj=0.7119 base=0.7906 mixed_floor=0.4000 align=0.9167
- run 2 seed=79 eps=0.22 solver_w=0.0 history_w=0.1 obj=0.7119 base=0.7906 mixed_floor=0.4000 align=0.9167
- run 3 seed=79 eps=0.22 solver_w=0.15 history_w=0.0 obj=0.7119 base=0.7906 mixed_floor=0.4000 align=0.9167
- run 4 seed=79 eps=0.22 solver_w=0.15 history_w=0.1 obj=0.7119 base=0.7906 mixed_floor=0.4000 align=0.9167

## Solver-tuning trajectory
```
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
*   *   *   *   *    *   *   *   *   *    *   *   *   *   *    *
x:[1, 16] y:[0.753, 1.753]
```
Final tuned objective=0.7534 baseline_avg=0.8316 mixed_floor=0.5833 solver_alignment=0.8750

## Policy selection frontier
- self_play/policy_final.json: objective=0.7262 baseline_avg=0.8060 mixed_floor=0.5625 solver_alignment=0.8333
- self_play_hof/policy_final.json: objective=0.7248 baseline_avg=0.7773 mixed_floor=0.5000 solver_alignment=1.0000
- run_002_seed_79_sw_0.00_hw_0.10_eps_0.22/policy_final.json: objective=0.7204 baseline_avg=0.7930 mixed_floor=0.5625 solver_alignment=0.8333
- run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json: objective=0.7122 baseline_avg=0.7747 mixed_floor=0.5625 solver_alignment=0.8333
- run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: objective=0.7089 baseline_avg=0.7917 mixed_floor=0.5312 solver_alignment=0.8333
- sweep_seed_11/policy_final.json: objective=0.6962 baseline_avg=0.7878 mixed_floor=0.5000 solver_alignment=0.8333

## Policy selection top-league Elo
```
run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json ################################################ 1208.890
sweep_seed_11/policy_final.json ############################################### 1208.360
run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json ############################################### 1207.930
run_002_seed_79_sw_0.00_hw_0.10_eps_0.22/policy_final.json ############################################### 1203.050
self_play/policy_final.json ############################################### 1188.820
self_play_hof/policy_final.json ############################################## 1182.950
```

## Milestone checklist
- Passed 12/13 milestones
  - [PASS] random_bot_runs: metric=1.0000 threshold=1.0000 (wins=9/13 draws=8)
  - [PASS] lower_life_totals_behavior: metric=0.7667 threshold=0.5500 (mixed_vs_random life=5)
  - [PASS] fixed_bolt_bot: metric=0.9000 threshold=0.5500 (bolt-only deck vs random)
  - [PASS] fixed_goblin_bot: metric=1.0000 threshold=0.5200 (goblin-only deck vs random)
  - [PASS] fixed_mixed_bot: metric=0.8000 threshold=0.5500 (mixed baseline vs random)
  - [PASS] trained_beats_random: metric=0.9000 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_bolt: metric=0.7500 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_goblin: metric=0.9500 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_mixed: metric=0.5000 threshold=0.5000 (life=10 hand=7)
  - [FAIL] trained_generalizes_life_hand: metric=0.4875 threshold=0.5000 (min winrate vs mixed over life∈{5,10,20}, hand∈{5,7})
  - [PASS] deck_better_than_starter: metric=0.5300 threshold=0.5000 (min best-deck winrate vs starter)
  - [PASS] elo_improves_over_training: metric=5.6244 threshold=0.0000 (last checkpoint Elo minus first checkpoint Elo)
  - [PASS] solver_alignment: metric=0.9167 threshold=0.8000 (agreement on sampled minimax-optimal states)

## Checkpoint convergence analysis
- Mean Elo delta for latest checkpoints: +3.11 (positive means later checkpoints still improving)
```
                         *                                     *
                                                                
                                                                
                                                                
                                                                
                                                                
            *                                     *             
                                                                
                                                                
                                                                
                                                                
*                                    *                          
x:[1, 6] y:[1.500, 4.610]
```
- checkpoint_ep20 -> checkpoint_ep25: win_rate_from=0.4375 elo_delta_to=1.66
- checkpoint_ep25 -> checkpoint_ep30: win_rate_from=0.3750 elo_delta_to=3.06
- checkpoint_ep30 -> checkpoint_ep35: win_rate_from=0.3125 elo_delta_to=4.61

## Recommended champion policy
- solver_tune/policy_best.json objective=0.6917 baseline_avg=0.8125 mixed_floor=0.5000 solver_alignment=0.8333

## Champion candidate leaderboard
- solver_tune/policy_best.json: robust=0.6917 baseline=0.8125 mixed_floor=0.5000 align=0.8333
- sweep_seed_11/policy_final.json: robust=0.6896 baseline=0.7656 mixed_floor=0.5000 align=0.9167
- sweep_seed_23/policy_final.json: robust=0.6865 baseline=0.7578 mixed_floor=0.5000 align=0.9167
- run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json: robust=0.6635 baseline=0.7630 mixed_floor=0.4375 align=0.9167
- self_play_hof/policy_final.json: robust=0.6510 baseline=0.7526 mixed_floor=0.3750 align=1.0000

## Head-to-head significance among top candidates
- Significant pairwise outcomes: 0/10
- solver_tune/policy_best.json vs sweep_seed_11/policy_final.json: wr_a=0.4500 CI=[0.2582, 0.6579] favored=sweep_seed_11/policy_final.json significant=0
- solver_tune/policy_best.json vs sweep_seed_23/policy_final.json: wr_a=0.4000 CI=[0.2188, 0.6134] favored=sweep_seed_23/policy_final.json significant=0
- solver_tune/policy_best.json vs run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json: wr_a=0.6000 CI=[0.3866, 0.7812] favored=solver_tune/policy_best.json significant=0
- solver_tune/policy_best.json vs self_play_hof/policy_final.json: wr_a=0.5000 CI=[0.2993, 0.7007] favored=none significant=0
- sweep_seed_11/policy_final.json vs sweep_seed_23/policy_final.json: wr_a=0.4000 CI=[0.2188, 0.6134] favored=sweep_seed_23/policy_final.json significant=0
- sweep_seed_11/policy_final.json vs run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json: wr_a=0.4000 CI=[0.2188, 0.6134] favored=run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json significant=0

## Replay sample index
- champion_vs_mixed_l10_h7: winner=1 result=player_1_wins turns=5 actions=107 replay=artifacts/replays/samples/champion_vs_mixed_l10_h7.jsonl
- champion_vs_bolt_l10_h7: winner=1 result=player_1_wins turns=5 actions=97 replay=artifacts/replays/samples/champion_vs_bolt_l10_h7.jsonl
- champion_vs_random_l10_h7: winner=0 result=player_0_wins turns=5 actions=114 replay=artifacts/replays/samples/champion_vs_random_l10_h7.jsonl
- champion_vs_goblin_l5_h5: winner=0 result=player_0_wins turns=4 actions=66 replay=artifacts/replays/samples/champion_vs_goblin_l5_h5.jsonl
- mixed_vs_bolt_l10_h7: winner=0 result=player_0_wins turns=6 actions=114 replay=artifacts/replays/samples/mixed_vs_bolt_l10_h7.jsonl
- mixed_vs_goblin_l10_h7: winner=0 result=player_0_wins turns=5 actions=111 replay=artifacts/replays/samples/mixed_vs_goblin_l10_h7.jsonl

## Artifact manifest
- git_head: 9b02403ff36d7248f5bf94fa38d1e5d27c0daf3f
- created_at_utc: 2026-02-23T05:05:07.397698+00:00
- recorded artifacts: 11 (present: 11)
  - artifacts/eval/champion.json bytes=260 sha256=18a092db942277ec...
  - artifacts/eval/champion_selection.csv bytes=2156 sha256=b6745da5d5352850...
  - artifacts/eval/generalization_matrix_tuned.csv bytes=815 sha256=0061d4df06526fa4...
  - artifacts/deck_search/results_tuned.csv bytes=272 sha256=f2658e84fae7fc5e...
  - artifacts/eval/deck_vs_starter_tuned.csv bytes=250 sha256=8a093a3a79481ecd...
  - artifacts/eval/policy_selection.csv bytes=2149 sha256=52c7ea62c8261db5...
  - artifacts/eval/policy_selection_matrix.csv bytes=2022 sha256=37490fab1c8c79d1...
  - artifacts/eval/significance.csv bytes=1509 sha256=a3d849c5ace72744...

## Optimality audit
- champion: solver_tune/policy_best.json
- solver_alignment=0.8333 best_baseline_alignment=1.0000 solver_gap=0.1667
- convergence_trailing_mean_elo_delta=+3.1100 objective_margin=0.0021
- checks: solver_gap=0 convergence=0 margin=0
- confidence=0.3858

## Healthcheck summary
- Passed checks: 4/6
  - [FAIL] milestones_all_passed: 12/13 passed
  - [PASS] champion_has_positive_objective: robust_objective=0.6917
  - [PASS] significance_matrix_nonempty: pairs=10 significant=0
  - [PASS] convergence_signal_within_bound: trailing_mean_elo_delta=+3.110
  - [FAIL] optimality_audit_strong: checks=0/3 confidence=0.3858 (thresholds: checks>=1, confidence>=0.35)
  - [PASS] report_contains_required_sections: all present