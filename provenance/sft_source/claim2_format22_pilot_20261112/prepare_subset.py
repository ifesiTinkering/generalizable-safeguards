import collections,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
SOURCE=Path('training/safepyramid_experiments/selection_v2/splits/defamation_to_content_moderation/sft/rows.parquet')
SELECTION=Path('training/safepyramid_experiments/production_v11/runs/defamation_to_content_moderation_thinking_off_correctness_calibration_sft/sft_selection.json')
rows=pq.read_table(SOURCE).to_pylist(); prior=json.loads(SELECTION.read_text())['rows']
assert len(rows)==100 and {r['row_id'] for r in rows}==set(prior)
chosen=[]
for flag,counts in [(True,[7,7,6]),(False,[1,1,0])]:
 for level,n in zip(['L0','L1','L2'],counts):
  group=[r for r in rows if bool(r['is_violation'])==flag and r['native_level']==level]
  group.sort(key=lambda r:hashlib.sha256(('claim2-format22-v1:'+r['row_id']).encode()).hexdigest())
  assert len(group)>=n
  chosen.extend(group[:n])
chosen.sort(key=lambda r:r['row_id']);assert len(chosen)==22 and len({r['row_id'] for r in chosen})==22
raw=json.dumps(chosen,indent=2,ensure_ascii=False,default=str)+'\n';(ROOT/'rows.json').write_text(raw)
manifest={'selection_rule':'Within label/native-level strata choose smallest SHA256(claim2-format22-v1:row_id); no model outcomes','source_parquet':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'original_selection_sha256':hashlib.sha256(SELECTION.read_bytes()).hexdigest(),'rows_sha256':hashlib.sha256(raw.encode()).hexdigest(),'row_ids':[r['row_id'] for r in chosen],'n':22,'positive':20,'negative':2,'native_level_counts':dict(collections.Counter(r['native_level'] for r in chosen)),'unique_conversations':len({r['conversation_sha256'] for r in chosen}),'target_domain':'defamation only, inherited original100 pool','recipe':{'initialization':'fresh raw base model, not b28','stage':'direct current four-tag structural-only SFT; no original JSON answer-supervised stage','epochs':3,'lr':.00001,'batching':'deterministic shuffle batch4 with2-row tail, everyrow once/pass;6updates/epoch18total','row_exposures':66},'validation_only':'same frozen90/current sharedv9prompt; report format validity separately from semantic F1; no heldout selection'}
(ROOT/'subset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ['n','positive','negative','native_level_counts','unique_conversations','rows_sha256']}))
