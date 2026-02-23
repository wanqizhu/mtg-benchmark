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
- life=5 hand=5: M=4 B=7 G=1 score=0.8438
- life=5 hand=7: M=8 B=4 G=0 score=0.8750
- life=10 hand=5: M=4 B=5 G=3 score=1.0000
- life=10 hand=7: M=5 B=6 G=1 score=0.9375
- life=20 hand=5: M=4 B=4 G=4 score=0.9219
- life=20 hand=7: M=4 B=5 G=3 score=1.0000

## Best deck vs starter validation
- life=5 hand=5 deck=M4/B7/G1 win_rate=0.6200 (62-38-0)
- life=5 hand=7 deck=M8/B4/G0 win_rate=0.5600 (56-44-0)
- life=10 hand=5 deck=M4/B5/G3 win_rate=0.6400 (64-36-0)
- life=10 hand=7 deck=M5/B6/G1 win_rate=0.5500 (55-45-0)
- life=20 hand=5 deck=M4/B4/G4 win_rate=0.5550 (54-43-3)
- life=20 hand=7 deck=M4/B5/G3 win_rate=0.7300 (73-27-0)

## Generalization matrix (trained vs baselines)
- life=5 hand=5 vs random: win_rate=0.8000 (96-24-0)
- life=5 hand=5 vs bolt: win_rate=0.6250 (75-45-0)
- life=5 hand=5 vs goblin: win_rate=0.9333 (112-8-0)
- life=5 hand=5 vs mixed: win_rate=0.4833 (58-62-0)
- life=5 hand=7 vs random: win_rate=0.7833 (94-26-0)
- life=5 hand=7 vs bolt: win_rate=0.5667 (68-52-0)
- life=5 hand=7 vs goblin: win_rate=0.9833 (118-2-0)
- life=5 hand=7 vs mixed: win_rate=0.5000 (60-60-0)
- life=10 hand=5 vs random: win_rate=0.8792 (105-14-1)
- life=10 hand=5 vs bolt: win_rate=0.8708 (104-15-1)
- life=10 hand=5 vs goblin: win_rate=0.9417 (113-7-0)
- life=10 hand=5 vs mixed: win_rate=0.5583 (67-53-0)
- life=10 hand=7 vs random: win_rate=0.9333 (112-8-0)
- life=10 hand=7 vs bolt: win_rate=0.8250 (99-21-0)
- life=10 hand=7 vs goblin: win_rate=0.9750 (117-3-0)
- life=10 hand=7 vs mixed: win_rate=0.5083 (61-59-0)
- life=20 hand=5 vs random: win_rate=0.6792 (43-0-77)
- life=20 hand=5 vs bolt: win_rate=0.8667 (88-0-32)
- life=20 hand=5 vs goblin: win_rate=0.8208 (86-9-25)
- life=20 hand=5 vs mixed: win_rate=0.6125 (63-36-21)
- life=20 hand=7 vs random: win_rate=0.7625 (64-1-55)
- life=20 hand=7 vs bolt: win_rate=0.9625 (112-1-7)
- life=20 hand=7 vs goblin: win_rate=0.9500 (113-5-2)
- life=20 hand=7 vs mixed: win_rate=0.6708 (79-38-3)

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
- sweep_seed_17/policy_final.json: objective=0.7303 baseline_avg=0.8151 mixed_floor=0.5625 solver_alignment=0.8333
- self_play_multi/policy_final.json: objective=0.7000 baseline_avg=0.7708 mixed_floor=0.4375 solver_alignment=1.0000
- sweep_seed_13/policy_final.json: objective=0.6994 baseline_avg=0.7695 mixed_floor=0.4375 solver_alignment=1.0000
- solver_tune/policy_best.json: objective=0.6885 baseline_avg=0.7708 mixed_floor=0.5000 solver_alignment=0.8333
- self_play_hof/policy_final.json: objective=0.6777 baseline_avg=0.7214 mixed_floor=0.4375 solver_alignment=1.0000
- sweep_seed_19/policy_final.json: objective=0.6734 baseline_avg=0.7604 mixed_floor=0.3750 solver_alignment=1.0000

## Policy selection top-league Elo
```
sweep_seed_17/policy_final.json ################################################ 1213.910
sweep_seed_13/policy_final.json ############################################### 1205.930
self_play_hof/policy_final.json ############################################### 1198.180
self_play_multi/policy_final.json ############################################### 1196.820
sweep_seed_19/policy_final.json ############################################### 1196.730
solver_tune/policy_best.json ############################################## 1188.430
```

## Milestone checklist
- Passed 10/13 milestones
  - [PASS] random_bot_runs: metric=1.0000 threshold=1.0000 (wins=14/8 draws=8)
  - [PASS] lower_life_totals_behavior: metric=0.7333 threshold=0.5500 (mixed_vs_random life=5)
  - [PASS] fixed_bolt_bot: metric=0.8833 threshold=0.5500 (bolt-only deck vs random)
  - [PASS] fixed_goblin_bot: metric=0.9083 threshold=0.5200 (goblin-only deck vs random)
  - [PASS] fixed_mixed_bot: metric=0.9167 threshold=0.5500 (mixed baseline vs random)
  - [PASS] trained_beats_random: metric=0.9583 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_bolt: metric=0.7833 threshold=0.5000 (life=10 hand=7)
  - [PASS] trained_beats_goblin: metric=0.9833 threshold=0.5000 (life=10 hand=7)
  - [FAIL] trained_beats_mixed: metric=0.4833 threshold=0.5000 (life=10 hand=7)
  - [FAIL] trained_generalizes_life_hand: metric=0.4875 threshold=0.5000 (min winrate vs mixed over life∈{5,10,20}, hand∈{5,7})
  - [PASS] deck_better_than_starter: metric=0.5500 threshold=0.5000 (min best-deck winrate vs starter)
  - [FAIL] elo_improves_over_training: metric=-11.2541 threshold=0.0000 (last checkpoint Elo minus first checkpoint Elo)
  - [PASS] solver_alignment: metric=0.9167 threshold=0.8000 (agreement on sampled minimax-optimal states)

## Checkpoint convergence analysis
- Mean Elo delta for latest checkpoints: +0.38 (positive means later checkpoints still improving)
```
                                                               *
                                                                
                                                                
                                                                
*                                                               
                                                                
                                                                
                                                                
            *                        *                          
                                                                
                         *                                      
                                                  *             
x:[1, 6] y:[-1.200, 2.360]
```
- checkpoint_ep20 -> checkpoint_ep25: win_rate_from=0.5000 elo_delta_to=-0.02
- checkpoint_ep25 -> checkpoint_ep30: win_rate_from=0.5500 elo_delta_to=-1.20
- checkpoint_ep30 -> checkpoint_ep35: win_rate_from=0.4000 elo_delta_to=2.36

## Recommended champion policy
- self_play_multi/policy_final.json objective=0.6983 baseline_avg=0.7958 mixed_floor=0.4500 solver_alignment=1.0000

## Champion candidate leaderboard
- self_play_multi/policy_final.json: robust=0.6983 baseline=0.7958 mixed_floor=0.4500 align=1.0000
- self_play/policy_final.json: robust=0.6946 baseline=0.8198 mixed_floor=0.5000 align=0.8333
- self_play_solver/policy_final.json: robust=0.6575 baseline=0.7438 mixed_floor=0.4000 align=1.0000
- self_play_hof/policy_final.json: robust=0.6254 baseline=0.7135 mixed_floor=0.3500 align=1.0000
- run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: robust=0.6242 baseline=0.7854 mixed_floor=0.4000 align=0.7500

## Head-to-head significance among top candidates
- Significant pairwise outcomes: 2/10
- self_play_multi/policy_final.json vs self_play/policy_final.json: wr_a=0.5375 CI=[0.4290, 0.6425] favored=self_play_multi/policy_final.json significant=0
- self_play_multi/policy_final.json vs self_play_solver/policy_final.json: wr_a=0.6125 CI=[0.5029, 0.7118] favored=self_play_multi/policy_final.json significant=1
- self_play_multi/policy_final.json vs self_play_hof/policy_final.json: wr_a=0.5125 CI=[0.4049, 0.6189] favored=self_play_multi/policy_final.json significant=0
- self_play_multi/policy_final.json vs run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json: wr_a=0.4125 CI=[0.3111, 0.5220] favored=run_003_seed_79_sw_0.15_hw_0.00_eps_0.22/policy_final.json significant=0
- self_play/policy_final.json vs self_play_solver/policy_final.json: wr_a=0.6375 CI=[0.5281, 0.7343] favored=self_play/policy_final.json significant=1
- self_play/policy_final.json vs self_play_hof/policy_final.json: wr_a=0.4875 CI=[0.3811, 0.5951] favored=self_play_hof/policy_final.json significant=0

## Replay sample index
- champion_vs_mixed_l10_h7: winner=1 result=player_1_wins turns=5 actions=114 replay=artifacts/replays/samples/champion_vs_mixed_l10_h7.jsonl
- champion_vs_bolt_l10_h7: winner=1 result=player_1_wins turns=6 actions=118 replay=artifacts/replays/samples/champion_vs_bolt_l10_h7.jsonl
- champion_vs_random_l10_h7: winner=0 result=player_0_wins turns=5 actions=108 replay=artifacts/replays/samples/champion_vs_random_l10_h7.jsonl
- champion_vs_goblin_l5_h5: winner=0 result=player_0_wins turns=4 actions=77 replay=artifacts/replays/samples/champion_vs_goblin_l5_h5.jsonl
- mixed_vs_bolt_l10_h7: winner=0 result=player_0_wins turns=6 actions=127 replay=artifacts/replays/samples/mixed_vs_bolt_l10_h7.jsonl
- mixed_vs_goblin_l10_h7: winner=0 result=player_0_wins turns=5 actions=111 replay=artifacts/replays/samples/mixed_vs_goblin_l10_h7.jsonl