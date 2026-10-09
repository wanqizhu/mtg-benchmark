import json,re,unicodedata
from pathlib import Path
from rapidfuzz import fuzz,process
p=Path('work/historical-calibration')
reviews=[('MSH','JM3Lz3tu7jY','cu'),('MSH','y11-rxC1rQk','rare'),('SOS','bsBoYVrt-mI','cu'),('SOS','Y1ESx1xg6fo','rare'),('ECL','xGEAEjCH8VE','cu'),('ECL','mxysznkjMB4','rare'),('TLA','jcAXn71UdaQ','cu')]
def norm(s):return re.sub(r'[^a-z0-9 ]','',unicodedata.normalize('NFKD',s).lower()).replace('  ',' ')
for set_,v,kind in reviews:
 caps=json.loads((p/'captions'/f'{v}.json').read_text())
 cards=json.loads((p/'tiers'/f'alexnikolic_{set_}_metadata.json').read_text())['ratings']
 cards=[x for x in cards if x['inclusion_type']=='Main Set' and x['rarity'] in (['C','U'] if kind=='cu' else ['R','M']) and 'basic' not in x['types']]
 windows=[norm(' '.join(x['text'] for x in caps[i:i+5])) for i in range(len(caps))]
 out=[]
 for card in cards:
  name=card['name'];candidates=[]
  for text,score,i in process.extract(norm(name.split(' // ')[0]),windows,scorer=fuzz.partial_ratio,limit=100):
   if caps[i]['start']<450:continue
   nearby=' '.join(x['text'] for x in caps[max(0,i-2):i+9]).lower()
   alignment=fuzz.partial_ratio_alignment(norm(name.split(' // ')[0]),text)
   pre=text[max(0,alignment.dest_start-50):alignment.dest_start]
   post=text[alignment.dest_end:alignment.dest_end+90]
   bonus=0
   if re.search(r'next (?:up )?(?:is|we have|weve got)?\s*$',pre):bonus+=26
   if re.search(r'^\s*(?:this|it|which) is\b',post):bonus+=24
   elif re.search(r'^\s*(?:is )?(?:one|two|three|four|five|six|seven|eight|nine|ten|white|blue|black|red|green)\b',post):bonus+=15
   if re.search(r'(?:give|grade| go | like | for | on ).{0,15}$',pre):bonus-=20
   if re.search(r'at (?:un)?common|at rare|at mythic',post):bonus+=4
   offset=0;matched_index=i
   for k in range(i,min(i+5,len(caps))):
    offset+=len(norm(caps[k]['text']))+1
    if offset>alignment.dest_start:
     matched_index=k;break
   candidates.append((score+bonus,score,caps[matched_index]['start'],matched_index,nearby))
  candidates.sort(reverse=True)
  best=candidates[0]
  # Earliest caption in the same winning introduction cluster.
  cluster=[x for x in candidates if abs(x[2]-best[2])<3 and x[0]>=best[0]-3]
  start=min(x[2] for x in cluster)
  out.append({'set':set_,'video':v,'kind':kind,'name':name,'rarity':card['rarity'],'color':card['color'],'start':start,'alignment_score':round(best[1],1),'candidate_score':round(best[0],1),'intro':best[4],'alternatives':[{ 'score':x[0],'start':x[2],'text':x[4]} for x in candidates[:3]]})
 out.sort(key=lambda x:x['start'])
 for i,x in enumerate(out):
  end=out[i+1]['start'] if i+1<len(out) else caps[-1]['start']+caps[-1]['duration']
  text=' '.join(z['text'] for z in caps if x['start']<=z['start']<end)
  x['end']=end;x['segment']=text
  grade_re=r'(?<![A-Za-z])(?:[BCDF](?:\+\+|\+|-| (?:plus|minus))?|A(?:\+|-| (?:plus|minus))?)(?![A-Za-z])'
  hits=[]
  for m in re.finditer(grade_re,text):
   if m.group()=='A' and not re.search('grade|give|go |like|bomb',text[max(0,m.start()-60):m.end()+60],re.I):continue
   hits.append(text[max(0,m.start()-55):min(len(text),m.end()+100)])
  x['grade_phrases']=hits
 (p/'captions'/f'{set_}_{kind}_aligned.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print(set_,kind,len(out),'low confidence',sum(x['alignment_score']<82 or x['end']-x['start']<15 for x in out),flush=True)
