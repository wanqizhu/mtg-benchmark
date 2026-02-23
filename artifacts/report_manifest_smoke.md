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
trained                  ################################################ 1229.212
mixed_bolt_face          ############################################### 1223.068
goblin_aggro             ############################################## 1186.515
random                   ############################################## 1185.835
bolt_face                ############################################# 1175.370
```

## Checkpoint league progression
```
                                          *                     
                                                                
                                                                
                                                                
                                                                
                     *         *                                
                                                                
                                                                
                                                                
                                                                
*                                                   *           
          *                                                    *
x:[5, 35] y:[1195.809, 1207.540]
```

## Deck search summary
- life=5 hand=5: M=5 B=4 G=3 score=0.8438
- life=5 hand=7: M=5 B=6 G=1 score=0.8438
- life=10 hand=5: M=4 B=5 G=3 score=1.0000
- life=10 hand=7: M=5 B=4 G=3 score=0.9062
- life=20 hand=5: M=3 B=9 G=0 score=0.9844
- life=20 hand=7: M=4 B=7 G=1 score=1.0000

## Best deck vs starter validation
- life=5 hand=5 deck=M5/B4/G3 win_rate=0.6400 (64-36-0)
- life=5 hand=7 deck=M5/B6/G1 win_rate=0.5800 (58-42-0)
- life=10 hand=5 deck=M4/B5/G3 win_rate=0.6100 (61-39-0)
- life=10 hand=7 deck=M5/B4/G3 win_rate=0.5100 (51-49-0)
- life=20 hand=5 deck=M3/B9/G0 win_rate=0.6850 (68-31-1)
- life=20 hand=7 deck=M4/B7/G1 win_rate=0.7300 (73-27-0)

## Generalization matrix (trained vs baselines)
- life=5 hand=5 vs random: win_rate=0.7667 (92-28-0)
- life=5 hand=5 vs bolt: win_rate=0.6167 (74-46-0)
- life=5 hand=5 vs goblin: win_rate=0.9250 (111-9-0)
- life=5 hand=5 vs mixed: win_rate=0.4833 (58-62-0)
- life=5 hand=7 vs random: win_rate=0.7750 (93-27-0)
- life=5 hand=7 vs bolt: win_rate=0.5583 (67-53-0)
- life=5 hand=7 vs goblin: win_rate=0.9750 (117-3-0)
- life=5 hand=7 vs mixed: win_rate=0.5083 (61-59-0)
- life=10 hand=5 vs random: win_rate=0.8792 (105-14-1)
- life=10 hand=5 vs bolt: win_rate=0.8875 (106-13-1)
- life=10 hand=5 vs goblin: win_rate=0.9500 (114-6-0)
- life=10 hand=5 vs mixed: win_rate=0.5667 (68-52-0)
- life=10 hand=7 vs random: win_rate=0.9333 (112-8-0)
- life=10 hand=7 vs bolt: win_rate=0.8333 (100-20-0)
- life=10 hand=7 vs goblin: win_rate=0.9750 (117-3-0)
- life=10 hand=7 vs mixed: win_rate=0.5250 (63-57-0)
- life=20 hand=5 vs random: win_rate=0.6833 (44-0-76)
- life=20 hand=5 vs bolt: win_rate=0.8583 (86-0-34)
- life=20 hand=5 vs goblin: win_rate=0.8083 (84-10-26)
- life=20 hand=5 vs mixed: win_rate=0.6000 (61-37-22)
- life=20 hand=7 vs random: win_rate=0.7542 (62-1-57)
- life=20 hand=7 vs bolt: win_rate=0.9625 (112-1-7)
- life=20 hand=7 vs goblin: win_rate=0.9292 (110-7-3)
- life=20 hand=7 vs mixed: win_rate=0.6417 (75-41-4)

## Decision differences on sampled states
- State 1: step=precombat_main P0 life=10 P1 life=10
  - trained: play_land
  - mixed: play_land
  - bolt: play_land
  - goblin: play_land
- State 2: step=precombat_main P0 life=10 P1 life=10
  - trained: cast_goblin
  - mixed: cast_bolt_face
  - bolt: cast_bolt_face
  - goblin: cast_goblin
- State 3: step=declare_attackers P0 life=10 P1 life=10
  - trained: declare_attackers_1
  - mixed: declare_attackers_1
  - bolt: declare_attackers_1
  - goblin: declare_attackers_1
- State 4: step=upkeep P0 life=10 P1 life=9
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
- sweep_seed_17/policy_final.json: objective=0.7395 baseline_avg=0.8099 mixed_floor=0.5000 solver_alignment=1.0000
- run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json: objective=0.6823 baseline_avg=0.7812 mixed_floor=0.4688 solver_alignment=0.8333
- self_play_hof/policy_final.json: objective=0.6645 baseline_avg=0.7161 mixed_floor=0.4062 solver_alignment=1.0000
- run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: objective=0.6567 baseline_avg=0.7487 mixed_floor=0.4375 solver_alignment=0.8333
- sweep_seed_13/policy_final.json: objective=0.6539 baseline_avg=0.7656 mixed_floor=0.3125 solver_alignment=1.0000
- sweep_seed_11/policy_final.json: objective=0.6536 baseline_avg=0.7904 mixed_floor=0.3750 solver_alignment=0.8333

## Policy selection top-league Elo
```
sweep_seed_17/policy_final.json ################################################ 1211.790
sweep_seed_13/policy_final.json ############################################### 1208.580
self_play_hof/policy_final.json ############################################### 1199.220
sweep_seed_11/policy_final.json ############################################### 1197.580
run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json ############################################### 1193.910
run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json ############################################### 1188.930
```

## Milestone checklist
- Passed 11/13 milestones
  - [PASS] random_bot_runs: metric=1.0000 threshold=1.0000 (wins=14/10 draws=6)
  - [PASS] lower_life_totals_behavior: metric=0.7500 threshold=0.5500 (mixed_vs_random life=5)
  - [PASS] fixed_bolt_bot: metric=0.8500 threshold=0.5500 (bolt-only deck vs random)
  - [PASS] fixed_goblin_bot: metric=0.8750 threshold=0.5200 (goblin-only deck vs random)
  - [PASS] fixed_mixed_bot: metric=0.9500 threshold=0.5500 (mixed baseline vs random)
  - [PASS] trained_beats_random: metric=0.9750 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_bolt: metric=0.8500 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_goblin: metric=1.0000 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_mixed: metric=0.5500 threshold=0.5000 (life=10 hand=7)
  - [FAIL] trained_generalizes_life_hand: metric=0.4875 threshold=0.5000 (min winrate vs mixed over life∈{5,10,20}, hand∈{5,7})
  - [PASS] deck_better_than_starter: metric=0.5100 threshold=0.5000 (min best-deck winrate vs starter)
  - [FAIL] elo_improves_over_training: metric=-12.2742 threshold=0.0000 (last checkpoint Elo minus first checkpoint Elo)
  - [PASS] solver_alignment: metric=0.9167 threshold=0.8000 (agreement on sampled minimax-optimal states)

## Checkpoint convergence analysis
- Mean Elo delta for latest checkpoints: -0.55 (positive means later checkpoints still improving)
```
                                                               *
                                                                
                                                                
*                                                               
                         *                                      
                                     *                          
                                                                
            *                                                   
                                                                
                                                                
                                                                
                                                  *             
x:[1, 6] y:[-6.000, 4.290]
```
- checkpoint_ep20 -> checkpoint_ep25: win_rate_from=0.5000 elo_delta_to=0.05
- checkpoint_ep25 -> checkpoint_ep30: win_rate_from=0.7500 elo_delta_to=-6.00
- checkpoint_ep30 -> checkpoint_ep35: win_rate_from=0.3125 elo_delta_to=4.29

## Recommended champion policy
- run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json objective=0.6500 baseline_avg=0.7917 mixed_floor=0.3750 solver_alignment=0.9167

## Champion candidate leaderboard
- run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json: robust=0.6500 baseline=0.7917 mixed_floor=0.3750 align=0.9167
- run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json: robust=0.6240 baseline=0.7682 mixed_floor=0.3750 align=0.8333
- sweep_seed_13/policy_final.json: robust=0.6146 baseline=0.7448 mixed_floor=0.3750 align=0.8333
- solver_tune/policy_best.json: robust=0.6125 baseline=0.7812 mixed_floor=0.3750 align=0.7500
- sweep_seed_17/policy_final.json: robust=0.6010 baseline=0.7526 mixed_floor=0.2500 align=1.0000

## Head-to-head significance among top candidates
- Significant pairwise outcomes: 0/10
- run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json vs run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json: wr_a=0.4000 CI=[0.2188, 0.6134] favored=run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json significant=0
- run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json vs sweep_seed_13/policy_final.json: wr_a=0.6500 CI=[0.4329, 0.8188] favored=run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json significant=0
- run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json vs solver_tune/policy_best.json: wr_a=0.6500 CI=[0.4329, 0.8188] favored=run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json significant=0
- run_001_seed_79_sw_0.00_hw_0.00_eps_0.22/policy_final.json vs sweep_seed_17/policy_final.json: wr_a=0.5000 CI=[0.2993, 0.7007] favored=none significant=0
- run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json vs sweep_seed_13/policy_final.json: wr_a=0.6000 CI=[0.3866, 0.7812] favored=run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json significant=0
- run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json vs solver_tune/policy_best.json: wr_a=0.5000 CI=[0.2993, 0.7007] favored=none significant=0

## Replay sample index
- champion_vs_mixed_l10_h7: winner=0 result=player_0_wins turns=5 actions=97 replay=artifacts/replays/samples/champion_vs_mixed_l10_h7.jsonl
- champion_vs_bolt_l10_h7: winner=0 result=player_0_wins turns=6 actions=118 replay=artifacts/replays/samples/champion_vs_bolt_l10_h7.jsonl
- champion_vs_random_l10_h7: winner=0 result=player_0_wins turns=6 actions=126 replay=artifacts/replays/samples/champion_vs_random_l10_h7.jsonl
- champion_vs_goblin_l5_h5: winner=0 result=player_0_wins turns=3 actions=44 replay=artifacts/replays/samples/champion_vs_goblin_l5_h5.jsonl
- mixed_vs_bolt_l10_h7: winner=1 result=player_1_wins turns=5 actions=99 replay=artifacts/replays/samples/mixed_vs_bolt_l10_h7.jsonl
- mixed_vs_goblin_l10_h7: winner=0 result=player_0_wins turns=6 actions=117 replay=artifacts/replays/samples/mixed_vs_goblin_l10_h7.jsonl

## Artifact manifest
- git_head: 34faea18324390169267f2992280fa76a12ad778
- created_at_utc: 2026-02-23T04:53:35.395814+00:00
- recorded artifacts: 11 (present: 11)
  - artifacts/eval/champion.json bytes=331 sha256=f21ef9168b98d38a...
  - artifacts/eval/champion_selection.csv bytes=2156 sha256=eb2c22e68ee648f6...
  - artifacts/eval/generalization_matrix_tuned.csv bytes=816 sha256=560d3a996d36b4a0...
  - artifacts/deck_search/results_tuned.csv bytes=272 sha256=864f41e345340b71...
  - artifacts/eval/deck_vs_starter_tuned.csv bytes=250 sha256=9badb069b0a61ddb...
  - artifacts/eval/policy_selection.csv bytes=2149 sha256=3dc655247d8cfb77...
  - artifacts/eval/policy_selection_matrix.csv bytes=1884 sha256=648223c1bdb3cc39...
  - artifacts/eval/significance.csv bytes=1623 sha256=f689332a735571e1...