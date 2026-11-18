"""Reproduce conservative V2 from preserved V1, without any new draws."""
import gzip,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent;P=V.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads(gzip.decompress((P/'RESULTS.json.gz').read_bytes()))
am=json.loads((V/'AMENDMENT.json').read_text())
assert sha(P/'RESULTS.json.gz')==am['prior_results_sha256']
for r in d['results']:
 for t in r['tests'].values():
  t['available']=bool(t['available'] and t['defined_bootstrap']>=.95*1999 and t['defined_null']>=.95*1999)
  if not t['available']:t['raw_p']=1.
s=[t for r in d['results']for t in r['tests'].values()];assert len(s)==1008
o=sorted(range(1008),key=lambda i:s[i]['raw_p']);v=1.
for rank in range(1007,-1,-1):
 i=o[rank];v=min(v,s[i]['raw_p']*1008/(rank+1));s[i]['BH_q']=v
v=0.
for rank,i in enumerate(o):
 v=max(v,min(1.,s[i]['raw_p']*(1008-rank)));s[i]['Holm_p']=v
d['schema']='direct-correctness-secondary1008-results-conservative-v2'
d['amendment_sha256']=sha(V/'AMENDMENT.json')
d['prior_results_sha256']=sha(P/'RESULTS.json.gz');d['new_draws']=0
raw=gzip.compress(json.dumps(d,sort_keys=True,allow_nan=False).encode(),mtime=0)
assert raw==(V/'RESULTS.json.gz').read_bytes()
print(json.dumps({'passed':True,'results_sha256':hashlib.sha256(raw).hexdigest(),'new_draws':0}))
