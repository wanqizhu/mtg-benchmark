import json
from pathlib import Path
p=Path('work/historical-calibration');dest=Path('work/fra-week-one-analysis/evaluations/historical-creators-2026-10-09/inputs')
src=Path('work/fra-week-one-analysis/evaluations/fra-week-one-2026-10-07/inputs/creator_ratings_native.json')
d=json.loads(src.read_text());keep=['casualspike','alexnikolic','marcanderson','limitedresources','draftsim'];sources=[x for x in d['sources'] if x['id'] in keep]
def norm(s):return s.split(' // ')[0].casefold().replace('’',"'")
for source in sources:
 files=[]
 if source['id']=='marcanderson':files=['marcanderson_FRA_metadata.json']
 elif source['id']=='alexnikolic':files=['alexnikolic_original_FRA_metadata.json','alexnikolic_current_FRA_metadata.json']
 lookups=[]
 for f in files:
  data=json.loads((p/'tiers'/f).read_text());assert data['last_updated']<'2026-09-29'
  lookups.append({norm(x['name']):x for x in data['ratings'] if x['tier'] not in [None,'TBD']})
 for row in source['rows']:
  # Use original CU publisher list when available; verified prelaunch full list for rare icons.
  r=next((lookup[norm(row['name'])] for lookup in lookups if norm(row['name']) in lookup),None)
  if r:
   seen={x['kind'] for x in row.get('contexts',[])}
   for flag,kind in [('buildaround','build_around_flag'),('synergy','synergy')]:
    if r.get('flags',{}).get(flag) and kind not in seen and not (kind=='build_around_flag' and 'build_around' in seen):
     row.setdefault('contexts',[]).append({'kind':kind,'native':None,'condition':'explicit prelaunch 17Lands icon','source':'prelaunch original CU / complete rare list'})
   row['context_metadata_note']='Enriched with explicit prelaunch publisher icons; raw grade unchanged from October 7 retained source'
lr=next(s for s in sources if s['id']=='limitedresources');seen={r['name'] for r in lr['rows']}
for r in json.loads((dest/'fra_qualitative_reviews.json').read_text()):
 if r['name'] not in seen:lr['rows'].append(r)
(dest/'fra_ratings.json').write_text(json.dumps({'sources':sources,'basis':'Original retained FRA raw grades, enriched only with explicit prelaunch 17Lands icons for Alex and Marc. Outcome data is not loaded by this step.'},ensure_ascii=False,indent=2))
print('sources',[(s['id'],len(s['rows'])) for s in sources])
