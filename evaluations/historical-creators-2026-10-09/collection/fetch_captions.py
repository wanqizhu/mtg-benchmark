import json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from youtube_transcript_api import YouTubeTranscriptApi
p=Path('work/historical-calibration/captions');p.mkdir(exist_ok=True)
ids=['JM3Lz3tu7jY','y11-rxC1rQk','Y1ESx1xg6fo','xGEAEjCH8VE','mxysznkjMB4','G9bqewk4i4Y','b-mJNcpSo-w']
def one(v):
 target=p/f'{v}.json'
 if target.exists(): return f'{v} cached'
 try:
  t=YouTubeTranscriptApi().fetch(v)
  data=json.dumps(t.to_raw_data(),ensure_ascii=False);target.write_text(data)
  log={'url':f'https://www.youtube.com/watch?v={v}','file':str(target),'retrieved_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sha256':hashlib.sha256(data.encode()).hexdigest(),'is_generated':t.is_generated,'language':t.language}
  (p/f'{v}.manifest.json').write_text(json.dumps(log,indent=2));return f'{v} {len(t)}'
 except Exception as e:return f'{v} ERROR {type(e).__name__} {str(e)[:250]}'
with ThreadPoolExecutor(max_workers=3) as ex:
 for s in ex.map(one,ids):print(s,flush=True)
