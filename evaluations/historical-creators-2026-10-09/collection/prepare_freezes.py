import json
from pathlib import Path
from datetime import datetime,timedelta
p=Path('work/historical-calibration')
ids=json.loads((p/'tier_ids.json').read_text());ids['casualspike']['ECL']='be15b13a5661422baa2953a68bebde20'
plan=[]
for creator,sets in ids.items():
 for set_,id_ in sets.items():
  d=json.loads((p/'tiers'/f'{creator}_{set_}_metadata.json').read_text())
  main={x['card_id'] for x in d['ratings'] if x['inclusion_type']=='Main Set' and x['tier'] not in [None,'TBD']}
  state={}
  for h in sorted(d['history'],key=lambda x:x['date']):
   for ch in h['changes']:state[ch['card_id']]=ch['to_tier']
   if all(state.get(i) not in [None,'TBD'] for i in main):
    at=(datetime.fromisoformat(h['date'])+timedelta(seconds=1)).isoformat()
    row={'creator':creator,'set':set_,'id':id_,'freeze_at':at,'last_updated':d['last_updated'],'main_rated_count':len(main),'grades':{str(i):state[i] for i in main}}
    plan.append(row);break
print('\n'.join(f"{x['creator']:14} {x['set']} {x['freeze_at']} {x['main_rated_count']}" for x in plan))
(p/'freeze_plan.json').write_text(json.dumps(plan,indent=2))
