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