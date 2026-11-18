"""Read-only saved-record rescoring; no client, judge request, or mutation."""
import sys,json,hashlib
from pathlib import Path
P=Path('/workspace/claim2-evidence-ablations-20261111/claim2_format22_pilot_20261112');R=P.parent/'runtime_v9_tinker_template';O=Path('/workspace/safepyramid-runs/claim2_evidence_ablations_20261111/partial/pilot_sft_learningcurve_20261112');sys.path[:0]=[str(P),str(R)]
from reward_pipeline import score_all
from heldout_metric_adapter import metric_values
from data.data_recipe import final_sampling_seed
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(P/'manifest.json');assert all(sha(Path(f))==h for f,h in m['source_hashes'].items());rows=read(P/'validation_rows.json');by={r['row_id']:r for r in rows};prompts=read(P/'validation_prompts.json');out={};sources={}
for condition in ['fresh_base','format12','format22','format32']:
 d=O/'evaluations/OFF'/condition;rs=read(d/'records.json');assert len(rs)==488 and len({r['row_id']for r in rs})==488 and {r['row_id']for r in rs}==set(by);repaired=[]
 for x in rs:
  row=by[x['row_id']];assert x['sampling_seed']==final_sampling_seed(row['split'],row['row_id']) and x['prompt_sha256']==hashlib.sha256(json.dumps(prompts['OFF'][x['row_id']]).encode()).hexdigest();assert x['sampling_config']=={'thinking':False,'temperature':1.0,'max_tokens':23000}
  unavailable=x.get('judge_unavailable',False);verdict=x.get('judge')
  def saved_judge(*unused):
   assert verdict is not None
   return verdict
  scores=score_all(row,x['text'],'partial',None if unavailable else saved_judge,x['clean_termination']);y=dict(x,**scores)
  if unavailable:y['judge_unavailable']=True
  repaired.append(y)
 scopes={'all':rows}
 for split in sorted({r['split']for r in rows}):scopes['split:'+split]=[r for r in rows if r['split']==split]
 domain=lambda r:r.get('domain')or r.get('target_domain')
 for dom in sorted({domain(r)for r in rows}):scopes['domain:'+dom]=[r for r in rows if domain(r)==dom]
 out[condition]={}
 for label,subset in scopes.items():
  ids={r['row_id']for r in subset};packet=metric_values([r for r in repaired if r['row_id']in ids],subset);assert len(packet['historical77_metric_names'])==77;out[condition][label]=packet
 sources[str(d/'records.json')]=sha(d/'records.json')
print(json.dumps({'scope':'OFF savedstudent outputs; correctedmissingjudge row parse; semantic remainsNone whereunavailable','manifest_sha256':sha(P/'manifest.json'),'source_sha256':sources,'cells':out,'provider_calls':0},allow_nan=False))
