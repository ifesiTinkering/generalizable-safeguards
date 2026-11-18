"""Deterministic disjoint source selection only; no training/evaluation calls."""
import hashlib,json,collections
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
SOURCE=Path('training/safepyramid_experiments/selection_v2/splits/defamation_to_content_moderation/sft/rows.parquet')
rows=pq.read_table(SOURCE).to_pylist();old=json.loads((ROOT/'rows32.json').read_text());oldids={x['row_id'] for x in old};assert len(oldids)==32
positive=[r for r in rows if r['is_violation'] and r['row_id'] not in oldids];assert len(positive)==20
key=lambda r:hashlib.sha256(('claim2-extension55-v1:'+r['row_id']).encode()).hexdigest()
negative=[]
for level in ['L0','L1','L2']:
 pool=[r for r in rows if not r['is_violation'] and r['native_level']==level and r['row_id'] not in oldids];negative.append(min(pool,key=key))
new=sorted(positive+negative,key=lambda r:r['row_id']);assert len(new)==23 and not(oldids&{r['row_id']for r in new})
combined=sorted(old+new,key=lambda r:r['row_id']);assert len(combined)==55 and sum(bool(r['is_violation'])for r in combined)==50
out=ROOT/'extension55';out.mkdir(exist_ok=True)
for name,data in [('new23_rows.json',new),('cumulative55_rows.json',combined)]:
 (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n')
manifest={'status':'PREPARED_SELECTION_ONLY_NOT_LAUNCHED','selection':'All20remaining original positive rows; one stable-hash-ranked unused negative pernativelevelL0/L1/L2. No outcomes used.','source_path':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'prior_rows32_sha256':hashlib.sha256((ROOT/'rows32.json').read_bytes()).hexdigest(),'new_count':23,'new_positive':20,'new_negative':3,'new_row_ids':[r['row_id']for r in new],'new_levels':dict(collections.Counter(r['native_level']for r in new)),'cumulative_count':55,'cumulative_positive':50,'cumulative_negative':5,'new23_sha256':hashlib.sha256((out/'new23_rows.json').read_bytes()).hexdigest(),'cumulative55_sha256':hashlib.sha256((out/'cumulative55_rows.json').read_bytes()).hexdigest(),'checkpoint_lineage':'Continue same optimizer from completed27-update32-row milestone; runtime scheduling remains owner/auditor decision','invariants':['No existing row repeated in NEW chunk','Old30positive milestone/output retained separately','Current prompt/parser unchanged','Parse-validity goal does not authorize parser relaxation','No forced F1 ceiling'],'provider_calls':0}
(out/'selection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
