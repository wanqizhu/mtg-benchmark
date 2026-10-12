import json,re,hashlib,unicodedata
from pathlib import Path
from collections import Counter
p=Path('work/historical-calibration');out=Path('work/fra-week-one-analysis/evaluations/historical-creators-2026-10-09/inputs')
release={'HOB':'2026-08-11','MSH':'2026-06-23','SOS':'2026-04-21','TMT':'2026-03-03','ECL':'2026-01-20','TLA':'2025-11-18'}
rows=[];ledger=[];exclusions=[]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(s):return unicodedata.normalize('NFKD',s.split(' // ')[0]).replace('’',"'").strip().casefold()
def row(creator,set_,name,native,ordinary=None,contexts=None,**extra):
 return {'creator':creator,'set':set_,'name':name,'native':native,'ordinary_native':ordinary if ordinary is not None else native,'contexts':contexts or [],**extra}
for f in json.loads((p/'freeze_plan.json').read_text()):
 c,s=f['creator'],f['set'];path=p/'tiers'/f'{c}_{s}_frozen.json';d=json.loads(path.read_text())
 url=f"https://www.17lands.com/data/tier_list/{f['id']}?date={f['freeze_at']}"
 for r in d['ratings']:
  if r['inclusion_type']!='Main Set' or r['tier'] in [None,'TBD'] or 'basic' in r['types']:continue
  ctx=[]
  for flag,kind in [('buildaround','build_around'),('synergy','synergy')]:
   if r.get('flags',{}).get(flag):ctx.append({'kind':kind,'native':None,'condition':'original tier-list icon'})
  if r['tier']=='SB':ctx.append({'kind':'sideboard','native':'SB','condition':'sideboard grade'})
  rows.append(row(c,s,r['name'],r['tier'],contexts=ctx,source_url=url,source_version=f['freeze_at'],evidence='17Lands historical version; grade replay checked'))
 ledger.append({'creator':c,'set':s,'admitted':True,'release':release[s],'published_or_frozen':f['freeze_at'],'basis':'17Lands earliest version covering all main cards that received a grade in this source; reconstructed from history','url':url,'archive_file':str(path),'archive_sha256':digest(path),'rated_main':sum(x['creator']==c and x['set']==s for x in rows),'complete':sum(x['inclusion_type']=='Main Set' and 'basic' not in x['types'] and x['tier'] not in [None,'TBD'] for x in d['ratings'])==sum(x['inclusion_type']=='Main Set' and 'basic' not in x['types'] for x in d['ratings']),'main_catalog_count':sum(x['inclusion_type']=='Main Set' and 'basic' not in x['types'] for x in d['ratings']),'notes':'Original version includes original flags; later grade and context edits are excluded.'})
# Original LR Hobbit transcription: full CU + rare episodes, source timestamps in every row.
hob=json.loads((p/'collector_HOB_aug14.json').read_text());vr=next(x for x in hob['videoRatings'] if x['sourceKey']=='limitedresources');anchors={norm(x['cardName']):x for x in vr['ratings']}
for card in hob['cards']:
 raw=card['ratings']['limitedresources']['native'];ordinary=raw;ctx=[]
 if 'build-around' in raw.lower():
  parts=re.split(r'\s*/\s*build-around\s*|^build-around\s*',raw,flags=re.I)
  ordinary=parts[0].strip() or None;cond=parts[-1].strip();ctx.append({'kind':'build_around','native':cond,'condition':'explicit build-around grade'})
  if ordinary is None:ordinary=cond
 for cond_text in re.findall(r'\(([^)]*)\)',raw):
  m=re.match(r'([A-F](?:\+\+|[+-])?(?:\s*/\s*[A-F][+-]?)?)\s+in\s+(.+)',cond_text)
  if m:ctx.append({'kind':'scenario','native':m.group(1),'condition':m.group(2)});ordinary=re.sub(r'\s*\([^)]*\)','',ordinary)
 anchor=anchors[norm(card['name'])];url=f"https://www.youtube.com/watch?v={anchor['videoId']}&t={anchor['timestampSeconds']}"
 rows.append(row('limitedresources','HOB',card['name'],raw,ordinary=ordinary,contexts=ctx,source_url=url,source_version=hob['generatedAt'],evidence='Dated secondary transcription, original video timestamp; raw C++ and ranges preserved'))
ledger.append({'creator':'limitedresources','set':'HOB','admitted':True,'release':release['HOB'],'published_or_frozen':'2026-08-14T18:12:49Z','basis':'Original Aug 3 CU / Aug 13 rare audio; Aug 4 / Aug 14 videos; self-dated Aug 14 full transcription','url':'https://chunk.science/mtga-hobbit.html','archive_file':str(p/'collector_HOB_aug14.json'),'archive_sha256':digest(p/'collector_HOB_aug14.json'),'rated_main':188,'complete':True,'notes':'Rare audio +2 days and video +3 days; early secondary file timestamp is recorded by publisher, not independently cryptographic proof.'})
manual=json.loads((p/'lr_manual.json').read_text())
for file in sorted((p/'captions').glob('*_direct.json')):
 key=file.stem.replace('_direct','');set_,kind=key.split('_');overrides=manual.get(key,{})
 for r in json.loads(file.read_text()):
  cfg=overrides.get(r['name'],{});hit=r['hits'][-1]
  if 'exclude' in cfg or cfg.get('note','').startswith('Front-face'):
   exclusions.append({'creator':'limitedresources','set':set_,'name':r['name'],'reason':cfg.get('exclude',cfg.get('note'))});continue
  grade=cfg.get('grade',hit['grade']);ctx=[]
  if cfg.get('context'):ctx=[{'kind':cfg['context'],'native':cfg.get('conditional_native',grade),'condition':'explicitly stated graded context in original captions'}]
  rows.append(row('limitedresources',set_,cfg.get('rename',r['name']),grade,contexts=ctx,grade_weights=cfg.get('grade_weights'),source_url=f"https://www.youtube.com/watch?v={r['video']}&t={int(hit['start'])}",source_version='original episode',evidence='Explicit spoken grade in auto captions; card attribution and raw grade manually checked; uncertain statements excluded'))
for r in manual['supplemental']:
 ctx=[]
 if r.get('context'):ctx=[{'kind':r['context'],'native':r.get('conditional_native',r['grade']),'condition':'explicit graded condition in original captions'}]
 rows.append(row('limitedresources',r['set'],r['name'],r['grade'],contexts=ctx,grade_weights=r.get('grade_weights'),source_url=f"https://www.youtube.com/watch?v={r['video']}&t={r['start']}",source_version='original episode',evidence=r.get('note','Explicit raw grade checked in continuous captions')))
for s,published in [('MSH','CU Jun 16 / rares Jun 25 (+2 days)'),('SOS','CU Apr 14 / rares Apr 21 (+0 days)'),('ECL','CU Jan 15 / rares Jan 22 (+2 days)'),('TLA','CU Nov 14; rare episode Nov 22 excluded (+4 days)')]:
 files=sorted((p/'captions').glob(f'{s}_*_direct.json'))
 ledger.append({'creator':'limitedresources','set':s,'admitted':True,'release':release[s],'published_or_frozen':published,'basis':'Dated original LR episodes, text-only reconstruction of explicit grades','url':'https://lrcast.com/episodes/','archive_files':[str(x) for x in files],'archive_sha256':{str(x):digest(x) for x in files},'rated_main':sum(x['creator']=='limitedresources' and x['set']==s for x in rows),'complete':False,'notes':'Partial caption reconstruction; unresolved card attribution or grade statements omitted. Includes rares where episode meets +3 day cutoff. TLA CU only; TMT primer is not a full graded review.'})
# Bot: frozen early Hobbit and unchanged SOS publication; retain decimal raw values.
bot=json.loads((p/'collector_HOB_aug11.json').read_text())
for card in bot['cards']:
 raw=card['ratings']['draftsim']['native'];rows.append(row('draftsim','HOB',card['name'],raw,source_url='https://draftsim.com/HOB-pick-order/',source_version=bot['generatedAt'],evidence='Original Aug 11 secondary snapshot; all 188 bot grades unchanged in Aug 14 copy'))
ledger.append({'creator':'draftsim','set':'HOB','admitted':True,'release':release['HOB'],'published_or_frozen':bot['generatedAt'],'basis':'Self-dated frozen Aug 11 bot snapshot, corroborated by unchanged Aug 14 values','url':'https://chunk.science/mtga-hobbit.html','archive_file':str(p/'collector_HOB_aug11.json'),'archive_sha256':digest(p/'collector_HOB_aug11.json'),'rated_main':188,'complete':True,'notes':'Secondary archive; later bot updates are excluded. Recorded source activity Aug 13 also lies within +3 day rule.'})
bot_main_names={norm(x['name']) for x in json.loads((p/'tiers/alexnikolic_SOS_metadata.json').read_text())['ratings'] if x['inclusion_type']=='Main Set' and 'basic' not in x['types']}
for line in (p/'draftsim_SOS_original.txt').read_text().splitlines()[1:]:
 a=line.split('\t')
 if len(a)<7 or not a[0]:continue
 name=a[0].replace('_',' ');raw=a[6]
 if norm(name) not in bot_main_names:continue
 rows.append(row('draftsim','SOS',name,raw,source_url='https://draftsim.com/SOS-pick-order/',source_version='2026-04-15',evidence='Public bot table labels latest update Apr 15; original native decimal rating from public data text'))
ledger.append({'creator':'draftsim','set':'SOS','admitted':True,'release':release['SOS'],'published_or_frozen':'2026-04-15','basis':'Publisher latest-update date predates release and no later edition evidence found','url':'https://draftsim.com/SOS-pick-order/','archive_file':str(p/'draftsim_SOS_original.txt'),'archive_sha256':digest(p/'draftsim_SOS_original.txt'),'rated_main':sum(x['creator']=='draftsim' and x['set']=='SOS' for x in rows),'complete':True,'notes':'Current public asset and display date; weaker timing guarantee than versioned 17Lands. Kept separately for source-strength sensitivity.'})
# Dataset assembly uses only published grade facts; outcome files are not loaded here.
counts=Counter((x['creator'],x['set'],norm(x['name'])) for x in rows);assert max(counts.values())==1, [k for k,n in counts.items() if n>1]
(out/'historical_ratings.json').write_text(json.dumps({'sources':ledger,'rows':rows,'extraction_exclusions':exclusions},ensure_ascii=False,indent=2))
for s,date in [('MSH','2026-07-13'),('TMT','2026-03-10'),('ECL','2026-02-17'),('TLA','2025-12-17'),('EOE','2025-08-26'),('TDM','2025-04-21'),('DFT','2025-02-28'),('FDN','2024-11-30'),('DSK','2024-10-12')]:
 ledger.append({'creator':'draftsim','set':s,'admitted':False,'published_or_frozen':date,'basis':'Current latest-update date after launch cutoff; no verified original archive recovered','url':f'https://draftsim.com/{s}-pick-order/','notes':'Written Andrew Quinn reviews are a different 0–10 source and are not substituted for bot grades.'})
ledger.extend([{'creator':'alexnikolic / marcanderson','set':'HOB','admitted':False,'basis':'Joint Alex + Marc list; cannot attribute to either individual model','url':'https://www.17lands.com/tier_list/528d1c45d1f04b59abac2a897a8928c8'},{'creator':'limitedresources','set':'TMT','admitted':False,'basis':'Only set primer located, not a complete native card-grade review'},{'creator':'limitedresources','set':'TLA rares','admitted':False,'published_or_frozen':'2025-11-22','basis':'Four days after Arena launch; exceeds declared three-day cutoff'},{'creator':'casualspike','set':'TLA / EOE','admitted':False,'basis':'Dated original authored posts located through public RSS; older than five most recent admitted Casual sets. Full rating-version recovery not needed after five-set cap.'}])
(out/'source_audit.json').write_text(json.dumps({'policy':'Original prediction editions no later than three calendar days after Arena launch; historical versions before subsequent updates are accepted. Main-set nonbasic cards only. No outcome-based source selection. Up to five sets per creator; missing sets are not invented.','arena_dates':release,'sources':ledger},ensure_ascii=False,indent=2))
print('rows',len(rows),'by creator/set',Counter((x['creator'],x['set']) for x in rows));print('LR exclusions',len(exclusions))
