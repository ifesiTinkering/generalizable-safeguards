"""Read-only completed-cell audit; no SDK/provider imports or calls."""
import sys,json,hashlib,math,collections
from pathlib import Path
CODE=Path('/workspace/claim2-evidence-ablations-20261111/claim2_format22_pilot_20261112')
RUNTIME=CODE.parent/'runtime_v9_tinker_template'
OUT=Path('/workspace/safepyramid-runs/claim2_evidence_ablations_20261111/partial/pilot_sft_learningcurve_20261112')
sys.path[:0]=[str(CODE),str(RUNTIME)]
from heldout_metric_adapter import metric_values
from reward_pipeline import score_all,parse
from data.data_recipe import final_sampling_seed
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(CODE/'manifest.json');assert all(sha(Path(p))==h for p,h in m['source_hashes'].items())
rows=read(CODE/'validation_rows.json');by={r['row_id']:r for r in rows};prompts=read(CODE/'validation_prompts.json');assert len(rows)==len(by)==488
report={'manifest_sha256':sha(CODE/'manifest.json'),'cells':{},'pending':[],'provider_calls':0}
def numeric_equal(a,b):
 if isinstance(a,dict):return set(a)==set(b)and all(numeric_equal(v,b[k])for k,v in a.items())
 if isinstance(a,list):return len(a)==len(b)and all(numeric_equal(x,y)for x,y in zip(a,b))
 if type(a)in(int,float)and type(b)in(int,float):return abs(a-b)<=1e-10
 return a==b
for mode in ['OFF','ON']:
 for condition in ['fresh_base','format12','format22','format32']:
  d=OUT/'evaluations'/mode/condition;label=mode+'/'+condition
  if not(d/'complete.json').exists():report['pending'].append(label);continue
  records=read(d/'records.json');packet=read(d/'metrics.json');done=read(d/'complete.json');assert done['records_sha256']==sha(d/'records.json')
  assert len(records)==488 and {r['row_id']for r in records}==set(by)
  valid=[];all_f1=[];all_c=[];all_p=[];all_r=[]
  for r in records:
   row=by[r['row_id']];expected={'thinking':mode=='ON','temperature':1.0,'max_tokens':10000 if mode=='ON'else 23000}
   assert r['sampling_config']==expected and r['sampling_seed']==final_sampling_seed(row['split'],row['row_id'])
   assert r['prompt_sha256']==hashlib.sha256(json.dumps(prompts[mode][r['row_id']]).encode()).hexdigest()
   assert 0<len(r['tokens'])<=expected['max_tokens'] and len(r['tokens'])==len(r['token_logprobs'])and all(math.isfinite(x)for x in r['token_logprobs'])
   assert r['decoder_audit']['sampled_token_count']==len(r['tokens'])and r['decoder_audit']['thinking_enabled']==expected['thinking']
   verdict=r.get('judge')
   def judge(*unused):
    assert verdict is not None,'Missing real judge verdict';return verdict
   recomputed=score_all(row,r['text'],'partial',judge,r['clean_termination'])
   for key,value in recomputed.items():assert numeric_equal(value,r[key]),(label,r['row_id'],key)
   if r['valid']:
    p=parse(r['text'],[str(x['rule_id'])for x in row['all_policy_rules']]);pred=set(p['violated_rule_ids']);gold=set(row['ground_truth_rule_ids']);tp=len(pred&gold)
    c=float(pred==gold);f1=2*tp/(len(pred)+len(gold))if pred or gold else 1.;precision=tp/len(pred)if pred else float(not gold);recall=tp/len(gold)if gold else float(not pred)
    valid.append((p['confidence'],c));all_f1.append(f1);all_c.append(c);all_p.append(precision);all_r.append(recall)
   else:all_f1.append(0.);all_c.append(0.);all_p.append(0.);all_r.append(0.)
  rebuilt=metric_values(records,rows);assert rebuilt['complete']and numeric_equal(rebuilt,packet)
  for key,values in [('macro_f1',all_f1),('exact_set_accuracy',all_c),('macro_precision',all_p),('macro_recall',all_r)]:assert abs(sum(values)/488-packet['stop_metrics'][key])<1e-10
  assert packet['populations']['parseable']==len(valid)
  if valid:assert abs(sum((q-c)**2 for q,c in valid)/len(valid)-packet['stop_metrics']['exact_correctness_brier'])<1e-10
  report['cells'][label]={'status':'PASS','records_sha256':sha(d/'records.json'),'metrics_sha256':sha(d/'metrics.json'),'rows':488,'parseable':len(valid),'metrics':packet['stop_metrics']}
print(json.dumps(report))
