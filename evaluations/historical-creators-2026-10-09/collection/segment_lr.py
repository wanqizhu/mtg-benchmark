import json,re,unicodedata,bisect
from pathlib import Path
from rapidfuzz import fuzz,process
p=Path('work/historical-calibration/captions')
def norm(s):return re.sub(r'[^a-z0-9 ]','',unicodedata.normalize('NFKD',s).lower()).replace('  ',' ')
for path in p.glob('*_aligned.json'):
 rows=json.loads(path.read_text());v=rows[0]['video'];caps=json.loads((p/f'{v}.json').read_text());full='';offsets=[]
 for x in caps:offsets.append(len(full));full+=norm(x['text'])+' '
 markers=[]
 names=[norm(x['name'].split(' // ')[0]) for x in rows]
 for m in re.finditer(r'\b(?:next up(?: is)?|next is|next we have|first up(?: is)?|our first card up is)\s+',full):
  excerpt=full[m.end():m.end()+130]
  matches=process.extract(excerpt,names,scorer=lambda x,y,**kw:fuzz.partial_ratio(y,x),limit=2)
  name_,score,index=matches[0];a=fuzz.partial_ratio_alignment(name_,excerpt)
  score-=max(0,a.dest_start-30)/4
  if score<73:continue
  markers.append({'start':caps[bisect.bisect_right(offsets,m.start())-1]['start'],'name':rows[index]['name'],'score':score,'excerpt':excerpt})
 # Deduplicate overlapping markers; preserve strongest explicitly introduced candidate per card.
 assignments={}
 for x in sorted(markers,key=lambda x:x['score'],reverse=True):
  if x['name'] not in assignments and not any(abs(x['start']-y['start'])<10 for y in assignments.values()):assignments[x['name']]=x
 # Recover introductions lacking a transition marker, and long names with difficult ASR.
 for r in sorted(rows,key=lambda x:x['candidate_score'],reverse=True):
  if r['name'] in assignments:continue
  start=r['start'];near=[x for x in markers if 0<=start-x['start']<180]
  if r['candidate_score']<120 and near: start=max(near,key=lambda x:x['start'])['start']
  if any(abs(start-y['start'])<15 for y in assignments.values()):continue
  if r['candidate_score']<105:continue
  assignments[r['name']]={'start':start,'name':r['name'],'score':r['candidate_score'],'excerpt':r['intro'],'recovered':True}
 ordered=sorted(assignments.values(),key=lambda x:x['start']);byname={x['name']:x for x in rows}
 for i,a in enumerate(ordered):
  r=byname[a['name']];end=ordered[i+1]['start'] if i+1<len(ordered) else caps[-1]['start']+caps[-1]['duration']
  text=' '.join(x['text'] for x in caps if a['start']<=x['start']<end)
  a.update({'set':r['set'],'video':v,'kind':r['kind'],'rarity':r['rarity'],'color':r['color'],'end':end,'segment':text})
  hits=[]
  for m in re.finditer(r'(?<![A-Za-z])(?:[BCDF](?:\+\+|\+|-| (?:plus|minus))?|A(?:\+|-| (?:plus|minus))?)(?![A-Za-z])',text):
   if m.group()=='A' and not re.search('grade|give|go |like|bomb',text[max(0,m.start()-60):m.end()+60],re.I):continue
   hits.append({'grade':m.group(),'at':m.start(),'snippet':text[max(0,m.start()-40):min(len(text),m.end()+75)]})
  a['grade_hits']=hits
 target=path.with_name(path.name.replace('_aligned','_segmented'));target.write_text(json.dumps(ordered,ensure_ascii=False,indent=2))
 print(path.name,len(ordered),'missing',len(rows)-len(ordered),flush=True)
