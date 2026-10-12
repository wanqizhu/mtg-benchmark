import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const root=path.resolve('work/historical-calibration');
fs.mkdirSync(root,{recursive:true});
export async function get(url,name){
  const target=path.join(root,name);fs.mkdirSync(path.dirname(target),{recursive:true});
  if(fs.existsSync(target))return fs.readFileSync(target,'utf8');
  const r=await fetch(url,{signal:AbortSignal.timeout(45000),headers:{'User-Agent':'mtg-benchmark-historical-calibration/1.0','Accept':'application/json,text/html,*/*'}});
  const text=await r.text();if(!r.ok)throw new Error(`${r.status} ${url}`);
  fs.writeFileSync(target,text);
  fs.appendFileSync(path.join(root,'downloads.jsonl'),JSON.stringify({url,file:name,retrieved_at:new Date().toISOString(),sha256:crypto.createHash('sha256').update(text).digest('hex'),bytes:Buffer.byteLength(text)})+'\n');
  return text;
}
if(process.argv[2]==='initial'){
  const jobs=[['https://www.17lands.com/card_tiers/data/420879087c5948b98bfe85ffc2985955','tiers/casualspike_SOS_current.json'],['https://chunk.science/mtga-reality-fracture.html','collector_FRA.html'],['https://limitedlevelups.com/tier-list/ECL','llu_ECL.html'],['https://api.github.com/repos/Toskicologist/MTG-draft-sets-infographics/contents','toski_files.json']];
  const r=await Promise.allSettled(jobs.map(async([u,n])=>{const t=await get(u,n);console.log(n,t.length,t.slice(0,60));}));
  for(const x of r)if(x.status==='rejected')console.log('ERROR',x.reason.message);
}
if(process.argv[2]==='sources'){
 const jobs=[
 ['https://limitedlevelups.com/assets/index-NeCOP4Li.js','llu_current.js'],
 ['https://chunk.science/mtga-reality-fracture/sets/registry.d3f0a25fa8da.js','collector_registry.js'],
 ['https://www.17lands.com/data/tier_list/420879087c5948b98bfe85ffc2985955','tiers/casualspike_SOS_metadata.json'],
 ['https://www.17lands.com/data/tier_list/528d1c45d1f04b59abac2a897a8928c8','tiers/llu_HOB_metadata.json'],
 ['https://raw.githubusercontent.com/Toskicologist/MTG-draft-sets-infographics/main/hob-ratings.html','toski_HOB_current.html'],
 ['https://api.github.com/repos/Toskicologist/MTG-draft-sets-infographics/commits?path=hob-ratings.html&per_page=100','toski_HOB_commits.json'],
 ];
 for(const set of ['HOB','MSH','SOS','TMT','ECL','TLA','EOE'])jobs.push([`https://www.17lands.com/api/card_data?expansion=${set}&event_type=PremierDraft&time_period=ALL_TIME`,`truth/${set}.json`]);
 for(let i=0;i<jobs.length;i+=3){await Promise.all(jobs.slice(i,i+3).map(async([u,n])=>{try{let t=await get(u,n);console.log(n,t.length);}catch(e){console.log('ERROR',n,e.message)}}));}
}
if(process.argv[2]==='archives'){
 const jobs=[
 ['https://chunk.science/mtga-hobbit.html','collector_HOB.html'],
 ['https://www.reddit.com/r/lrcast/comments/1q9j1r4/lorwyn_eclipsed_breakdown/','casual_ECL_post.html'],
 ['https://lrcast.com/episodes/page/3/','lr_episodes3.html'],
 ['https://lrcast.com/episodes/page/4/','lr_episodes4.html'],
 ['https://raw.githubusercontent.com/Toskicologist/MTG-draft-sets-infographics/main/ecl-ratings-v2.html','toski_ECL_current.html'],
 ['https://api.github.com/repos/Toskicologist/MTG-draft-sets-infographics/commits?path=ecl-ratings-v2.html&per_page=100','toski_ECL_commits.json']
 ];
 for(let i=0;i<jobs.length;i+=3){await Promise.all(jobs.slice(i,i+3).map(async([u,n])=>{try{let t=await get(u,n);console.log(n,t.length);}catch(e){console.log('ERROR',n,e.message)}}));}
}
if(process.argv[2]==='selected'){
 const ids={casualspike:{HOB:'aed3cb5bc0604f9db2cfd1ce93688cce',MSH:'4c638bcf3b134a40853cb5920e1d9328',SOS:'420879087c5948b98bfe85ffc2985955'},alexnikolic:{MSH:'b07c077b8c8145288f75d71bf4f90d65',SOS:'3dda4b67b4b0403caeb0a744887b0208',TMT:'c1489602e3d544d9aa90718692f16a32',ECL:'4f09f4891cc148f0bb7afe510d4605ec',TLA:'6554520225504ef9a3d5360c062a8274'},marcanderson:{MSH:'e0c4c50e90914ac390a1f792e0717ed2',SOS:'09ccc33bf4d6461f9c73de01bd8efe0c',TMT:'a8a502ee50d84a31b6a501bb6b4d5920',ECL:'0fef073a7db5401ba7e260e8649f2bea',TLA:'10ab962ac03b4774b7cd2cb8bec152af'}};
 fs.writeFileSync(path.join(root,'tier_ids.json'),JSON.stringify(ids,null,2));
 const jobs=[];for(const [creator,sets]of Object.entries(ids))for(const [set,id]of Object.entries(sets))jobs.push([`https://www.17lands.com/data/tier_list/${id}`,`tiers/${creator}_${set}_metadata.json`]);
 jobs.push(['https://chunk.science/mtga-hobbit/data/hob_versions.69972eb90519.js?v=20260927-source-refresh','collector_HOB_versions.js']);
 for(const set of ['HOB','MSH','SOS','TMT','ECL','TLA'])jobs.push([`https://draftsim.com/${set}-pick-order/`,`draftsim_${set}_current.html`]);
 for(let i=0;i<jobs.length;i+=2){await Promise.all(jobs.slice(i,i+2).map(async([u,n])=>{try{let t=await get(u,n);console.log(n,t.length);}catch(e){console.log('ERROR',n,e.message)}}));}
}
