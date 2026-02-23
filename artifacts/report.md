# MTG AI Report

## Training progression
### Win rate vs random over training
```
          **************************                            
                                                                
                                                                
                                                                
                                                                
*                                   ****************************
                                                                
 *********                                                      
                                                                
                                                                
                                                                
*                                                               
x:[1, 80] y:[0.750, 1.000]
```
### Win rate vs mixed baseline over training
```
*         ******************************************************
                                                                
                                                                
                                                                
 *********                                                      
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
*                                                               
x:[1, 80] y:[0.450, 0.750]
```
Final pseudo-Elo vs mixed baseline: 1390.85

## League Elo snapshot
```
trained                  ################################################ 1227.640
mixed_bolt_face          ############################################### 1224.276
goblin_aggro             ############################################## 1187.020
random                   ############################################## 1185.655
bolt_face                ############################################# 1175.409
```

## Deck search summary
- life=5 hand=5: M=5 B=6 G=1 score=0.8750
- life=5 hand=7: M=6 B=6 G=0 score=0.8125
- life=10 hand=5: M=4 B=8 G=0 score=0.9062
- life=10 hand=7: M=4 B=5 G=3 score=0.9688
- life=20 hand=5: M=3 B=5 G=4 score=0.9219
- life=20 hand=7: M=3 B=8 G=1 score=1.0000

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
- State 3: step=postcombat_main P0 life=10 P1 life=10
  - trained: cast_goblin
  - mixed: cast_bolt_face
  - bolt: cast_bolt_face
  - goblin: cast_goblin
- State 4: step=upkeep P0 life=7 P1 life=7
  - trained: cast_bolt_face
  - mixed: cast_bolt_face
  - bolt: cast_bolt_face
  - goblin: pass_priority