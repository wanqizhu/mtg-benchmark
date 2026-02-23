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
trained                  ################################################ 1229.414
mixed_bolt_face          ############################################### 1223.047
goblin_aggro             ############################################## 1186.701
random                   ############################################## 1185.635
bolt_face                ############################################# 1175.204
```

## Checkpoint league progression
```
                     *                                          
                                                                
*                                                               
                                                               *
          *                                                     
                                          *                     
                               *                                
                                                                
                                                                
                                                                
                                                                
                                                    *           
x:[5, 35] y:[1188.009, 1207.038]
```

## Deck search summary
- life=5 hand=5: M=6 B=4 G=2 score=0.8438
- life=5 hand=7: M=5 B=4 G=3 score=0.8125
- life=10 hand=5: M=5 B=7 G=0 score=0.9688
- life=10 hand=7: M=4 B=2 G=6 score=0.9375
- life=20 hand=5: M=3 B=6 G=3 score=0.9844
- life=20 hand=7: M=4 B=8 G=0 score=1.0000

## Best deck vs starter validation
- life=5 hand=5 deck=M6/B4/G2 win_rate=0.5800 (58-42-0)
- life=5 hand=7 deck=M5/B4/G3 win_rate=0.6100 (61-39-0)
- life=10 hand=5 deck=M5/B7/G0 win_rate=0.6000 (60-40-0)
- life=10 hand=7 deck=M4/B2/G6 win_rate=0.5100 (51-49-0)
- life=20 hand=5 deck=M3/B6/G3 win_rate=0.7200 (71-27-2)
- life=20 hand=7 deck=M4/B8/G0 win_rate=0.7900 (79-21-0)

## Generalization matrix (trained vs baselines)
- life=5 hand=5 vs random: win_rate=0.7750 (93-27-0)
- life=5 hand=5 vs bolt: win_rate=0.6250 (75-45-0)
- life=5 hand=5 vs goblin: win_rate=0.9333 (112-8-0)
- life=5 hand=5 vs mixed: win_rate=0.4833 (58-62-0)
- life=5 hand=7 vs random: win_rate=0.7750 (93-27-0)
- life=5 hand=7 vs bolt: win_rate=0.5750 (69-51-0)
- life=5 hand=7 vs goblin: win_rate=0.9833 (118-2-0)
- life=5 hand=7 vs mixed: win_rate=0.5250 (63-57-0)
- life=10 hand=5 vs random: win_rate=0.8792 (105-14-1)
- life=10 hand=5 vs bolt: win_rate=0.8875 (106-13-1)
- life=10 hand=5 vs goblin: win_rate=0.9417 (113-7-0)
- life=10 hand=5 vs mixed: win_rate=0.5833 (70-50-0)
- life=10 hand=7 vs random: win_rate=0.9417 (113-7-0)
- life=10 hand=7 vs bolt: win_rate=0.8500 (102-18-0)
- life=10 hand=7 vs goblin: win_rate=0.9750 (117-3-0)
- life=10 hand=7 vs mixed: win_rate=0.5333 (64-56-0)
- life=20 hand=5 vs random: win_rate=0.6792 (43-0-77)
- life=20 hand=5 vs bolt: win_rate=0.8583 (86-0-34)
- life=20 hand=5 vs goblin: win_rate=0.8208 (85-8-27)
- life=20 hand=5 vs mixed: win_rate=0.6042 (61-36-23)
- life=20 hand=7 vs random: win_rate=0.7458 (60-1-59)
- life=20 hand=7 vs bolt: win_rate=0.9542 (110-1-9)
- life=20 hand=7 vs goblin: win_rate=0.9250 (109-7-4)
- life=20 hand=7 vs mixed: win_rate=0.6333 (74-42-4)

## Decision differences on sampled states
- State 1: step=precombat_main P0 life=10 P1 life=10
  - trained: play_land
  - mixed: play_land
  - bolt: play_land
  - goblin: play_land
- State 2: step=precombat_main P0 life=10 P1 life=10
  - trained: cast_bolt_face
  - mixed: cast_bolt_face
  - bolt: cast_bolt_face
  - goblin: pass_priority
- State 3: step=upkeep P0 life=7 P1 life=7
  - trained: cast_bolt_face
  - mixed: cast_bolt_face
  - bolt: cast_bolt_face
  - goblin: pass_priority
- State 4: step=precombat_main P0 life=7 P1 life=4
  - trained: play_land
  - mixed: play_land
  - bolt: play_land
  - goblin: play_land

## Solver alignment in tractable low-life regime
- Compared states: 12
- Trained agreement: 0.833
- Mixed baseline agreement: 0.917
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
- self_play_hof/policy_final.json: objective=0.6959 baseline_avg=0.7617 mixed_floor=0.4375 solver_alignment=1.0000
- sweep_seed_17/policy_final.json: objective=0.6866 baseline_avg=0.8151 mixed_floor=0.4375 solver_alignment=0.8333
- sweep_seed_23/policy_final.json: objective=0.6784 baseline_avg=0.7969 mixed_floor=0.4375 solver_alignment=0.8333
- self_play_solver/policy_final.json: objective=0.6754 baseline_avg=0.7161 mixed_floor=0.4375 solver_alignment=1.0000
- run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json: objective=0.6714 baseline_avg=0.7812 mixed_floor=0.4375 solver_alignment=0.8333
- sweep_seed_19/policy_final.json: objective=0.6699 baseline_avg=0.7526 mixed_floor=0.3750 solver_alignment=1.0000

## Policy selection top-league Elo
```
run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json ################################################ 1216.350
sweep_seed_17/policy_final.json ############################################### 1208.610
sweep_seed_23/policy_final.json ############################################### 1195.110
self_play_solver/policy_final.json ############################################### 1194.680
self_play_hof/policy_final.json ############################################### 1193.910
sweep_seed_19/policy_final.json ############################################### 1191.340
```

## Milestone checklist
- Passed 10/13 milestones
  - [PASS] random_bot_runs: metric=1.0000 threshold=1.0000 (wins=8/13 draws=9)
  - [PASS] lower_life_totals_behavior: metric=0.7667 threshold=0.5500 (mixed_vs_random life=5)
  - [PASS] fixed_bolt_bot: metric=0.9000 threshold=0.5500 (bolt-only deck vs random)
  - [PASS] fixed_goblin_bot: metric=0.9500 threshold=0.5200 (goblin-only deck vs random)
  - [PASS] fixed_mixed_bot: metric=0.9000 threshold=0.5500 (mixed baseline vs random)
  - [PASS] trained_beats_random: metric=0.9500 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_bolt: metric=0.8500 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_goblin: metric=1.0000 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_mixed: metric=0.6500 threshold=0.5000 (life=10 hand=7)
  - [FAIL] trained_generalizes_life_hand: metric=0.4625 threshold=0.5000 (min winrate vs mixed over life∈{5,10,20}, hand∈{5,7})
  - [PASS] deck_better_than_starter: metric=0.5100 threshold=0.5000 (min best-deck winrate vs starter)
  - [FAIL] elo_improves_over_training: metric=-5.1814 threshold=0.0000 (last checkpoint Elo minus first checkpoint Elo)
  - [FAIL] solver_alignment: metric=0.7500 threshold=0.8000 (agreement on sampled minimax-optimal states)

## Checkpoint convergence analysis
- Mean Elo delta for latest checkpoints: -1.05 (positive means later checkpoints still improving)
```
*                                                               
                                                               *
                                                                
                         *                                      
                                                                
                                                                
            *                                                   
                                                                
                                                                
                                                                
                                                                
                                     *            *             
x:[1, 6] y:[-3.100, 3.000]
```
- checkpoint_ep20 -> checkpoint_ep25: win_rate_from=0.6250 elo_delta_to=-2.95
- checkpoint_ep25 -> checkpoint_ep30: win_rate_from=0.6250 elo_delta_to=-3.10
- checkpoint_ep30 -> checkpoint_ep35: win_rate_from=0.3750 elo_delta_to=2.89

## Recommended champion policy
- self_play/policy_final.json objective=0.7729 baseline_avg=0.8490 mixed_floor=0.6250 solver_alignment=0.9167

## Champion candidate leaderboard
- self_play/policy_final.json: robust=0.7729 baseline=0.8490 mixed_floor=0.6250 align=0.9167
- self_play_multi/policy_final.json: robust=0.7344 baseline=0.8359 mixed_floor=0.5000 align=1.0000
- solver_tune/policy_best.json: robust=0.7042 baseline=0.8021 mixed_floor=0.5000 align=0.9167
- run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: robust=0.7021 baseline=0.7969 mixed_floor=0.5000 align=0.9167
- run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json: robust=0.6833 baseline=0.7917 mixed_floor=0.5000 align=0.8333

## Head-to-head significance among top candidates
- Significant pairwise outcomes: 1/10
- self_play/policy_final.json vs self_play_multi/policy_final.json: wr_a=0.5000 CI=[0.2993, 0.7007] favored=none significant=0
- self_play/policy_final.json vs solver_tune/policy_best.json: wr_a=0.6500 CI=[0.4329, 0.8188] favored=self_play/policy_final.json significant=0
- self_play/policy_final.json vs run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: wr_a=0.7500 CI=[0.5313, 0.8881] favored=self_play/policy_final.json significant=1
- self_play/policy_final.json vs run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json: wr_a=0.4500 CI=[0.2582, 0.6579] favored=run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json significant=0
- self_play_multi/policy_final.json vs solver_tune/policy_best.json: wr_a=0.4000 CI=[0.2188, 0.6134] favored=solver_tune/policy_best.json significant=0
- self_play_multi/policy_final.json vs run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: wr_a=0.3500 CI=[0.1812, 0.5671] favored=run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json significant=0

## Replay sample index
- champion_vs_mixed_l10_h7: winner=1 result=player_1_wins turns=5 actions=99 replay=artifacts/replays/samples/champion_vs_mixed_l10_h7.jsonl
- champion_vs_bolt_l10_h7: winner=0 result=player_0_wins turns=5 actions=97 replay=artifacts/replays/samples/champion_vs_bolt_l10_h7.jsonl
- champion_vs_random_l10_h7: winner=0 result=player_0_wins turns=6 actions=117 replay=artifacts/replays/samples/champion_vs_random_l10_h7.jsonl
- champion_vs_goblin_l5_h5: winner=0 result=player_0_wins turns=3 actions=55 replay=artifacts/replays/samples/champion_vs_goblin_l5_h5.jsonl
- mixed_vs_bolt_l10_h7: winner=0 result=player_0_wins turns=5 actions=111 replay=artifacts/replays/samples/mixed_vs_bolt_l10_h7.jsonl
- mixed_vs_goblin_l10_h7: winner=0 result=player_0_wins turns=7 actions=133 replay=artifacts/replays/samples/mixed_vs_goblin_l10_h7.jsonl

## Artifact manifest
- git_head: 1c646c7ba02642441e11b94867451e823b92de6d
- created_at_utc: 2026-02-23T04:58:02.245348+00:00
- recorded artifacts: 11 (present: 11)
  - artifacts/eval/champion.json bytes=258 sha256=b79f0783ce000259...
  - artifacts/eval/champion_selection.csv bytes=2156 sha256=21a6fb040a5d54a8...
  - artifacts/eval/generalization_matrix_tuned.csv bytes=815 sha256=15edd791f57cdd0f...
  - artifacts/deck_search/results_tuned.csv bytes=272 sha256=e031bb2df25dae4a...
  - artifacts/eval/deck_vs_starter_tuned.csv bytes=250 sha256=5c7391b9fe9d1938...
  - artifacts/eval/policy_selection.csv bytes=2149 sha256=2662f5de619f05e7...
  - artifacts/eval/policy_selection_matrix.csv bytes=1740 sha256=7eee9b6d701a7ee4...
  - artifacts/eval/significance.csv bytes=1631 sha256=99db6b10807e5e63...

## Optimality audit
- champion: self_play/policy_final.json
- solver_alignment=0.6667 best_baseline_alignment=1.0000 solver_gap=0.3333
- convergence_trailing_mean_elo_delta=-1.0533 objective_margin=0.0385
- checks: solver_gap=0 convergence=1 margin=1
- confidence=0.7438