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
trained                  ################################################ 1229.795
mixed_bolt_face          ############################################### 1223.062
goblin_aggro             ############################################## 1187.239
random                   ############################################## 1185.262
bolt_face                ############################################# 1174.643
```

## Checkpoint league progression
```
                                          *                     
                                                                
                                                                
                     *                                          
                                                                
                               *                                
                                                                
                                                                
                                                                
          *                                                     
                                                               *
*                                                   *           
x:[5, 35] y:[1193.711, 1209.329]
```

## Deck search summary
- life=5 hand=5: M=4 B=4 G=4 score=0.8125
- life=5 hand=7: M=5 B=5 G=2 score=0.7812
- life=10 hand=5: M=4 B=6 G=2 score=0.9688
- life=10 hand=7: M=4 B=4 G=4 score=0.9375
- life=20 hand=5: M=3 B=6 G=3 score=0.9688
- life=20 hand=7: M=3 B=6 G=3 score=1.0000

## Best deck vs starter validation
- life=5 hand=5 deck=M4/B4/G4 win_rate=0.5800 (58-42-0)
- life=5 hand=7 deck=M5/B5/G2 win_rate=0.5500 (55-45-0)
- life=10 hand=5 deck=M4/B6/G2 win_rate=0.6000 (60-40-0)
- life=10 hand=7 deck=M4/B4/G4 win_rate=0.5500 (55-45-0)
- life=20 hand=5 deck=M3/B6/G3 win_rate=0.6600 (66-34-0)
- life=20 hand=7 deck=M3/B6/G3 win_rate=0.7500 (75-25-0)

## Generalization matrix (trained vs baselines)
- life=5 hand=5 vs random: win_rate=0.8500 (102-18-0)
- life=5 hand=5 vs bolt: win_rate=0.7500 (90-30-0)
- life=5 hand=5 vs goblin: win_rate=0.9417 (113-7-0)
- life=5 hand=5 vs mixed: win_rate=0.5833 (70-50-0)
- life=5 hand=7 vs random: win_rate=0.8667 (104-16-0)
- life=5 hand=7 vs bolt: win_rate=0.6750 (81-39-0)
- life=5 hand=7 vs goblin: win_rate=0.9917 (119-1-0)
- life=5 hand=7 vs mixed: win_rate=0.5750 (69-51-0)
- life=10 hand=5 vs random: win_rate=0.9500 (113-5-2)
- life=10 hand=5 vs bolt: win_rate=0.9083 (108-10-2)
- life=10 hand=5 vs goblin: win_rate=0.9333 (112-8-0)
- life=10 hand=5 vs mixed: win_rate=0.5750 (69-51-0)
- life=10 hand=7 vs random: win_rate=0.9625 (115-4-1)
- life=10 hand=7 vs bolt: win_rate=0.9167 (110-10-0)
- life=10 hand=7 vs goblin: win_rate=0.9750 (117-3-0)
- life=10 hand=7 vs mixed: win_rate=0.5833 (70-50-0)
- life=20 hand=5 vs random: win_rate=0.6500 (36-0-84)
- life=20 hand=5 vs bolt: win_rate=0.8583 (86-0-34)
- life=20 hand=5 vs goblin: win_rate=0.8375 (85-4-31)
- life=20 hand=5 vs mixed: win_rate=0.6042 (59-34-27)
- life=20 hand=7 vs random: win_rate=0.7750 (66-0-54)
- life=20 hand=7 vs bolt: win_rate=0.9625 (111-0-9)
- life=20 hand=7 vs goblin: win_rate=0.9292 (109-6-5)
- life=20 hand=7 vs mixed: win_rate=0.7083 (84-34-2)

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
- State 4: step=upkeep P0 life=10 P1 life=10
  - trained: cast_bolt_face
  - mixed: cast_bolt_face
  - bolt: cast_bolt_face
  - goblin: pass_priority

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
- run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: objective=0.7330 baseline_avg=0.7956 mixed_floor=0.5000 solver_alignment=1.0000
- run_002_seed_79_sw_0.00_hw_0.10_eps_0.22/policy_final.json: objective=0.7207 baseline_avg=0.7682 mixed_floor=0.5000 solver_alignment=1.0000
- self_play/policy_final.json: objective=0.7152 baseline_avg=0.8047 mixed_floor=0.4375 solver_alignment=1.0000
- sweep_seed_11/policy_final.json: objective=0.7006 baseline_avg=0.7721 mixed_floor=0.4375 solver_alignment=1.0000
- sweep_seed_19/policy_final.json: objective=0.6953 baseline_avg=0.7604 mixed_floor=0.4375 solver_alignment=1.0000
- run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json: objective=0.6846 baseline_avg=0.7852 mixed_floor=0.3750 solver_alignment=1.0000

## Policy selection top-league Elo
```
sweep_seed_11/policy_final.json ################################################ 1206.020
self_play/policy_final.json ############################################### 1205.260
run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json ############################################### 1202.810
sweep_seed_19/policy_final.json ############################################### 1199.910
run_004_seed_79_sw_0.15_hw_0.10_eps_0.22/policy_final.json ############################################### 1199.910
run_002_seed_79_sw_0.00_hw_0.10_eps_0.22/policy_final.json ############################################### 1186.100
```

## Milestone checklist
- Passed 13/13 milestones
  - [PASS] random_bot_runs: metric=1.0000 threshold=1.0000 (wins=12/9 draws=9)
  - [PASS] lower_life_totals_behavior: metric=0.7500 threshold=0.5500 (mixed_vs_random life=5)
  - [PASS] fixed_bolt_bot: metric=0.8750 threshold=0.5500 (bolt-only deck vs random)
  - [PASS] fixed_goblin_bot: metric=0.9208 threshold=0.5200 (goblin-only deck vs random)
  - [PASS] fixed_mixed_bot: metric=0.8542 threshold=0.5500 (mixed baseline vs random)
  - [PASS] trained_beats_random: metric=0.9250 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_bolt: metric=0.8417 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_goblin: metric=0.9750 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_mixed: metric=0.5167 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_generalizes_life_hand: metric=0.5500 threshold=0.5000 (min winrate vs mixed over life∈{5,10,20}, hand∈{5,7})
  - [PASS] deck_better_than_starter: metric=0.5500 threshold=0.5000 (min best-deck winrate vs starter)
  - [PASS] elo_improves_over_training: metric=8.0930 threshold=0.0000 (last checkpoint Elo minus first checkpoint Elo)
  - [PASS] solver_alignment: metric=0.8333 threshold=0.8000 (agreement on sampled minimax-optimal states)

## Checkpoint convergence analysis
- Mean Elo delta for latest checkpoints: +2.67 (positive means later checkpoints still improving)
```
                                                  *             
                                                                
                                                                
                                                                
                                                                
                                                                
*                                    *                         *
                                                                
                                                                
                                                                
                                                                
            *            *                                      
x:[1, 6] y:[-1.140, 4.560]
```
- checkpoint_ep20 -> checkpoint_ep25: win_rate_from=0.4250 elo_delta_to=1.78
- checkpoint_ep25 -> checkpoint_ep30: win_rate_from=0.3125 elo_delta_to=4.56
- checkpoint_ep30 -> checkpoint_ep35: win_rate_from=0.4375 elo_delta_to=1.66

## Recommended champion policy
- self_play/policy_final.json objective=0.7133 baseline_avg=0.8000 mixed_floor=0.4833 solver_alignment=1.0000

## Champion candidate leaderboard
- self_play/policy_final.json: robust=0.7133 baseline=0.8000 mixed_floor=0.4833 align=1.0000
- solver_tune/policy_best.json: robust=0.7107 baseline=0.7934 mixed_floor=0.4833 align=1.0000
- self_play_multi/policy_final.json: robust=0.7099 baseline=0.8080 mixed_floor=0.5500 align=0.8333
- run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: robust=0.6968 baseline=0.7837 mixed_floor=0.5000 align=0.9167
- self_play_hof/policy_final.json: robust=0.6818 baseline=0.7378 mixed_floor=0.4667 align=1.0000

## Head-to-head significance among top candidates
- Significant pairwise outcomes: 2/10
- self_play/policy_final.json vs solver_tune/policy_best.json: wr_a=0.4833 CI=[0.4209, 0.5463] favored=solver_tune/policy_best.json significant=0
- self_play/policy_final.json vs self_play_multi/policy_final.json: wr_a=0.4750 CI=[0.4127, 0.5381] favored=self_play_multi/policy_final.json significant=0
- self_play/policy_final.json vs run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: wr_a=0.5125 CI=[0.4496, 0.5750] favored=self_play/policy_final.json significant=0
- self_play/policy_final.json vs self_play_hof/policy_final.json: wr_a=0.5667 CI=[0.5034, 0.6278] favored=self_play/policy_final.json significant=1
- solver_tune/policy_best.json vs self_play_multi/policy_final.json: wr_a=0.4792 CI=[0.4168, 0.5422] favored=self_play_multi/policy_final.json significant=0
- solver_tune/policy_best.json vs run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: wr_a=0.5125 CI=[0.4496, 0.5750] favored=solver_tune/policy_best.json significant=0

## Replay sample index
- champion_vs_mixed_l10_h7: winner=1 result=player_1_wins turns=5 actions=99 replay=artifacts/replays/samples/champion_vs_mixed_l10_h7.jsonl
- champion_vs_bolt_l10_h7: winner=1 result=player_1_wins turns=5 actions=99 replay=artifacts/replays/samples/champion_vs_bolt_l10_h7.jsonl
- champion_vs_random_l10_h7: winner=0 result=player_0_wins turns=5 actions=97 replay=artifacts/replays/samples/champion_vs_random_l10_h7.jsonl
- champion_vs_goblin_l5_h5: winner=0 result=player_0_wins turns=3 actions=57 replay=artifacts/replays/samples/champion_vs_goblin_l5_h5.jsonl
- mixed_vs_bolt_l10_h7: winner=0 result=player_0_wins turns=5 actions=96 replay=artifacts/replays/samples/mixed_vs_bolt_l10_h7.jsonl
- mixed_vs_goblin_l10_h7: winner=0 result=player_0_wins turns=5 actions=96 replay=artifacts/replays/samples/mixed_vs_goblin_l10_h7.jsonl