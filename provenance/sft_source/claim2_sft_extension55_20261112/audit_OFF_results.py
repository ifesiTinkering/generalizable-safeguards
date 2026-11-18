import sys,json,hashlib,math
from pathlib import Path
P=Path('/workspace/claim2-evidence-ablations-20261111/claim2_format22_pilot_20261112');R=P.parent/'runtime_v9_tinker_template';O=Path('/workspace/safepyramid-runs/claim2_evidence_ablations_20261111/partial/pilot_sft_learningcurve_20261112');sys.path[:0]=[str(P),str(R)]
from reward_pipeline import score_all,parse
from heldout_metric_adapter import metric_values
from data.data_recipe import final_sampling_seed
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
E=P.parent/'claim2_sft_extension55_20261112';em=read(E/'manifest.json');assert all(sha(Path(f))==h for f,h in em['source_hashes'].items());assert sha(E/'manifest.json')=='753a5bb28ed28d2a7d49cbfad3795f4ba1007fdb5cfb25494aecc0c56c458b2d'
m=read(P/'manifest.json');assert all(sha(Path(f))==h for f,h in m['source_hashes'].items());rows=read(P/'validation_rows.json');by={r['row_id']:r for r in rows};prompts=read(P/'validation_prompts.json');out={}
for condition in ['extension55']:
 d=O.parent/'pilot_sft_extension55_20261112/evaluations/OFF';rs=read(d/'records.json');assert len(rs)==488 and {r['row_id']for r in rs}==set(by);repaired=[];missing=[]
 for x in rs:
  row=by[x['row_id']];assert x['sampling_seed']==final_sampling_seed(row['split'],row['row_id']) and x['prompt_sha256']==hashlib.sha256(json.dumps(prompts['OFF'][x['row_id']]).encode()).hexdigest();assert x['sampling_config']=={'thinking':False,'temperature':1.0,'max_tokens':23000}
  assert 0<len(x['tokens'])<=23000 and len(x['tokens'])==len(x['token_logprobs'])and all(math.isfinite(z)for z in x['token_logprobs'])
  unavailable=x.get('judge_unavailable',False);verdict=x.get('judge')
  def judge(*unused):assert verdict is not None;return verdict
  scores=score_all(row,x['text'],'partial',None if unavailable else judge,x['clean_termination'])
  y=dict(x,**scores)
  if unavailable:missing.append({'row_id':x['row_id'],'reason':x.get('reason'),'reparsed_valid':scores['valid']});y['judge_unavailable']=True
  repaired.append(y)
 packet=metric_values(repaired,rows);assert not packet['complete'] and missing
 independent={k:[] for k in ['macro_f1','macro_precision','macro_recall','exact_set_accuracy']};briers=[]
 for y in repaired:
  row=by[y['row_id']]
  if not y['valid']:
   for vals in independent.values():vals.append(0.)
   continue
  p=parse(y['text'],[str(z['rule_id'])for z in row['all_policy_rules']]);pred=set(p['violated_rule_ids']);gold=set(row['ground_truth_rule_ids']);tp=len(pred&gold);c=float(pred==gold)
  independent['macro_f1'].append(2*tp/(len(pred)+len(gold))if pred or gold else 1.)
  independent['macro_precision'].append(tp/len(pred)if pred else float(not gold))
  independent['macro_recall'].append(tp/len(gold)if gold else float(not pred));independent['exact_set_accuracy'].append(c);briers.append((p['confidence']-c)**2)
 for key,vals in independent.items():assert len(vals)==488 and abs(sum(vals)/488-packet['stop_metrics'][key])<1e-10
 assert abs(sum(briers)/len(briers)-packet['stop_metrics']['exact_correctness_brier'])<1e-10
 core={k:v for k,v in packet['stop_metrics'].items()if not k.startswith('semantic_')}
 assert all(packet['stop_metrics'][k]is None for k in ['semantic_aggregate_evidence','semantic_per_rule_evidence'])
 out[condition]={'status':'ALL488_STUDENT_CORRECTNESS_VERIFIED_SEMANTIC_INCOMPLETE','rows':488,'missing_semantic':missing,'populations':packet['populations'],'nonsemantic_metrics':core,'semantic_metrics':None,'raw_records_sha256':sha(d/'records.json'),'original_metrics_sha256':sha(d/'metrics.json'),'warning':'Student fields independently re-scored from all saved text; missing semantic outcomes remain N/A. Original extension packet already preserves student correctness.'}
print(json.dumps({'manifest_sha256':sha(E/'manifest.json'),'parent_manifest_sha256':sha(P/'manifest.json'),'cells':out,'provider_calls':0}))
