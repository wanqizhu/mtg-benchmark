# MTG AI Report

## Training progression
### Win rate vs random over training
```
           ** * ** * ** ** * ** ** * ** * ** ** * ** ** * **    
                                                                
                                                                
                                                             * *
                                                                
                                                                
** **                                                           
                                                                
                                                                
      *                                                         
                                                                
        **                                                      
x:[1, 40] y:[0.562, 0.823]
```
### Win rate vs mixed baseline over training
```
                                                             * *
                                                                
                                                                
                                                                
                                                                
      *    ** * ** * ** ** * ** ** * ** * ** ** * ** ** * **    
                                                                
                                                                
                                                                
        **                                                      
                                                                
** **                                                           
x:[1, 40] y:[0.354, 0.656]
```
Final pseudo-Elo vs mixed baseline: 1312.29
Convergence check (fitness delta between recent windows): +0.0031

## League Elo snapshot
```
mixed_bolt_face          ################################################ 1229.994
trained                  ############################################## 1202.741
goblin_aggro             ############################################## 1194.075
random                   ############################################## 1192.449
bolt_face                ############################################## 1180.742
```

## Checkpoint league progression
```
                           *                                    
                                                                
                                    *                           
                                                                
                  *                          *                  
                                                                
                                                                
                                                                
                                                               *
                                                                
*                                                     *         
         *                                                      
x:[5, 40] y:[1192.192, 1208.456]
```

## Deck search summary
- life=5 hand=5: M=4 B=8 G=0 score=0.8750
- life=5 hand=7: M=2 B=8 G=2 score=0.8438
- life=10 hand=5: M=4 B=4 G=4 score=0.8750
- life=10 hand=7: M=5 B=5 G=2 score=0.9688
- life=20 hand=5: M=3 B=6 G=3 score=0.9219
- life=20 hand=7: M=4 B=4 G=4 score=0.9688

## Generalization matrix (trained vs baselines)
- life=5 hand=5 vs random: win_rate=0.7667 (92-28-0)
- life=5 hand=5 vs bolt: win_rate=0.6417 (77-43-0)
- life=5 hand=5 vs goblin: win_rate=0.9333 (112-8-0)
- life=5 hand=5 vs mixed: win_rate=0.5667 (68-52-0)
- life=5 hand=7 vs random: win_rate=0.7333 (88-32-0)
- life=5 hand=7 vs bolt: win_rate=0.6250 (75-45-0)
- life=5 hand=7 vs goblin: win_rate=0.9667 (116-4-0)
- life=5 hand=7 vs mixed: win_rate=0.5250 (63-57-0)
- life=10 hand=5 vs random: win_rate=0.9000 (107-11-2)
- life=10 hand=5 vs bolt: win_rate=0.9083 (109-11-0)
- life=10 hand=5 vs goblin: win_rate=0.9500 (114-6-0)
- life=10 hand=5 vs mixed: win_rate=0.5250 (63-57-0)
- life=10 hand=7 vs random: win_rate=0.8750 (105-15-0)
- life=10 hand=7 vs bolt: win_rate=0.8500 (102-18-0)
- life=10 hand=7 vs goblin: win_rate=0.9917 (119-1-0)
- life=10 hand=7 vs mixed: win_rate=0.5083 (61-59-0)
- life=20 hand=5 vs random: win_rate=0.6333 (32-0-88)
- life=20 hand=5 vs bolt: win_rate=0.8000 (72-0-48)
- life=20 hand=5 vs goblin: win_rate=0.7667 (71-7-42)
- life=20 hand=5 vs mixed: win_rate=0.5250 (42-36-42)
- life=20 hand=7 vs random: win_rate=0.7583 (62-0-58)
- life=20 hand=7 vs bolt: win_rate=0.9042 (98-1-21)
- life=20 hand=7 vs goblin: win_rate=0.8333 (92-12-16)
- life=20 hand=7 vs mixed: win_rate=0.5375 (61-52-7)

## Decision differences on sampled states
- State 1: step=precombat_main P0 life=10 P1 life=10
  - trained: play_land
  - mixed: play_land
  - bolt: play_land
  - goblin: play_land
- State 2: step=postcombat_main P0 life=10 P1 life=10
  - trained: play_land
  - mixed: play_land
  - bolt: play_land
  - goblin: play_land
- State 3: step=precombat_main P0 life=10 P1 life=10
  - trained: play_land
  - mixed: play_land
  - bolt: play_land
  - goblin: play_land
- State 4: step=postcombat_main P0 life=10 P1 life=10
  - trained: play_land
  - mixed: play_land
  - bolt: play_land
  - goblin: play_land

## Solver alignment in tractable low-life regime
- Compared states: 12
- Trained agreement: 0.833
- Mixed baseline agreement: 1.000
- Bolt baseline agreement: 1.000
- Goblin baseline agreement: 0.917