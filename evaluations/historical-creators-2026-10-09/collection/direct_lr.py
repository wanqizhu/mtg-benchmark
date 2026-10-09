import json,re,unicodedata,bisect
from pathlib import Path
from rapidfuzz import fuzz,process
p=Path('work/historical-calibration/captions')
def norm(s):return re.sub(r'[^a-z0-9 ]','',unicodedata.normalize('NFKD',s).lower()).replace('  ',' ')
reviews=[('MSH','JM3Lz3tu7jY','cu'),('MSH','y11-rxC1rQk','rare'),('SOS','bsBoYVrt-mI','cu'),('SOS','Y1ESx1xg6fo','rare'),('ECL','xGEAEjCH8VE','cu'),('ECL','mxysznkjMB4','rare'),('TLA','jcAXn71UdaQ','cu')]
for set_,v,kind in reviews:
 cards=json.loads((p/f'{set_}_{kind}_aligned.json').read_text());names=[norm(x['name'].split(' // ')[0]) for x in cards]
 caps=json.loads((p/f'{v}.json').read_text());text='';offsets=[]
 for x in caps:offsets.append(len(text));text+=x['text']+' '
 rows={x['name']:{'set':set_,'video':v,'kind':kind,'name':x['name'],'hits':[]} for x in cards}
 grade_re=r'(?<![A-Za-z])(?:[BCDF](?:\+\+|\+|-| (?:plus|minus))?|A(?:\+|-| (?:plus|minus))?)(?![A-Za-z])'
 for m in re.finditer(grade_re,text):
  if m.group()=='A':
   if not re.search(r'give|grade| go |like|bomb|an ',text[max(0,m.start()-40):m.start()],re.I):continue
  bef=text[max(0,m.start()-150):m.start()];aft=text[m.end():m.end()+150]
  targets=[]
  if re.match(r'\s*(?:for|on|to)\s+',aft):
   targets.append((re.sub(r'^\s*(?:for|on|to)\s+','',aft)[:100],0))
  # "Name looks like a grade" and similar explicit links.
  if re.search(r'looks|give|go|say|think|is|it|seems',bef[-60:],re.I):targets.append((bef[-120:],-7))
  for target,penalty in targets:
   matches=process.extract(norm(target),names,scorer=lambda x,y,**kw:fuzz.partial_ratio(y,x),limit=2)
   if not matches:continue
   name_,score,i=matches[0]
   alignment=fuzz.partial_ratio_alignment(name_,norm(target))
   if penalty==0 and alignment.dest_start>12:continue
   if penalty<0:
    gap=norm(target)[alignment.dest_end:]
    if len(gap)>65 or 'next' in gap:continue
   if score+penalty<84 or score-matches[1][1]<6:continue
   grade=m.group().replace(' plus','+').replace(' minus','-')
   time=caps[bisect.bisect_right(offsets,m.start())-1]['start']
   if time<450:continue
   rows[cards[i]['name']]['hits'].append({'grade':grade,'start':time,'match':round(score+penalty,1),'snippet':text[max(0,m.start()-90):m.end()+170]})
   break
 out=[x for x in rows.values() if x['hits']];out.sort(key=lambda x:x['hits'][-1]['start'])
 (p/f'{set_}_{kind}_direct.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(set_,kind,len(out),flush=True)
