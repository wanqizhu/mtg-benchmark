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
- life=5 hand=5: M=4 B=8 G=0 score=0.8750
- life=5 hand=7: M=2 B=8 G=2 score=0.8438
- life=10 hand=5: M=4 B=4 G=4 score=0.8750
- life=10 hand=7: M=5 B=5 G=2 score=0.9688
- life=20 hand=5: M=3 B=6 G=3 score=0.9219
- life=20 hand=7: M=4 B=4 G=4 score=0.9688

## Generalization matrix (trained vs baselines)
- life=5 hand=5 vs random: win_rate=0.9500 (114-6-0)
- life=5 hand=5 vs bolt: win_rate=0.6917 (83-37-0)
- life=5 hand=5 vs goblin: win_rate=0.9500 (114-6-0)
- life=5 hand=5 vs mixed: win_rate=0.5750 (69-51-0)
- life=5 hand=7 vs random: win_rate=0.9083 (109-11-0)
- life=5 hand=7 vs bolt: win_rate=0.6167 (74-46-0)
- life=5 hand=7 vs goblin: win_rate=0.9917 (119-1-0)
- life=5 hand=7 vs mixed: win_rate=0.5667 (68-52-0)
- life=10 hand=5 vs random: win_rate=0.9083 (108-10-2)
- life=10 hand=5 vs bolt: win_rate=0.9292 (111-8-1)
- life=10 hand=5 vs goblin: win_rate=0.9208 (110-9-1)
- life=10 hand=5 vs mixed: win_rate=0.5333 (64-56-0)
- life=10 hand=7 vs random: win_rate=0.9333 (112-8-0)
- life=10 hand=7 vs bolt: win_rate=0.8500 (102-18-0)
- life=10 hand=7 vs goblin: win_rate=0.9583 (115-5-0)
- life=10 hand=7 vs mixed: win_rate=0.5417 (65-55-0)
- life=20 hand=5 vs random: win_rate=0.6542 (37-0-83)
- life=20 hand=5 vs bolt: win_rate=0.8458 (83-0-37)
- life=20 hand=5 vs goblin: win_rate=0.8125 (81-6-33)
- life=20 hand=5 vs mixed: win_rate=0.6458 (65-30-25)
- life=20 hand=7 vs random: win_rate=0.7958 (72-1-47)
- life=20 hand=7 vs bolt: win_rate=0.9625 (111-0-9)
- life=20 hand=7 vs goblin: win_rate=0.9333 (108-4-8)
- life=20 hand=7 vs mixed: win_rate=0.7167 (84-32-4)

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