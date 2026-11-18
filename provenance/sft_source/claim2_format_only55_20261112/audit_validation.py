import sys,json,hashlib,math
from pathlib import Path
P=Path('/workspace/claim2-evidence-ablations-20261111/claim2_format_only55_20261112');sys.path.insert(0,str(P));import format_only as f
from reward_pipeline import parse
from data.data_recipe import training_sampling_seed
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();m=read(P/'manifest.json');assert all(sha(p)==h for p,h in m['source_hashes'].items());rows=read(P/'validation90_rows.json');by={r['row_id']:r for r in rows};prompts=read(P/'validation_prompts.json');out={}
for d in sorted((f.OUT/'validation').glob('epoch*')):
 if not(d/'complete.json').exists():continue
 done=read(d/'complete.json');rs=read(d/'records.json');assert len(rs)==90 and {r['row_id']for r in rs}==set(by);assert done['records_sha256']==sha(d/'records.json');vals=[]
 for r in rs:
  row=by[r['row_id']];assert r['sampling_seed']==training_sampling_seed(0,'validation',r['row_id'],0);assert r['prompt_sha256']==hashlib.sha256(json.dumps(prompts[r['row_id']]).encode()).hexdigest();assert r['sampling_config']=={'thinking':False,'temperature':1.0,'max_tokens':23000}
  try:
   assert r['clean_termination'];p=parse(r['text'],[str(x['rule_id'])for x in row['all_policy_rules']]);pred=set(p['violated_rule_ids']);gold=set(row['ground_truth_rule_ids']);tp=len(pred&gold);v=(1,2*tp/(len(pred)+len(gold))if pred or gold else 1.,tp/len(pred)if pred else float(not gold),tp/len(gold)if gold else float(not pred),float(pred==gold))
  except (ValueError,AssertionError):v=(0,0.,0.,0.,0.)
  assert bool(v[0])==r['valid'];assert abs(v[1]-r['T'])<1e-12;vals.append(v)
 assert sum(v[0]for v in vals)==done['valid'];out[d.name]={'n':90,'valid':sum(v[0]for v in vals),'validity':sum(v[0]for v in vals)/90,'macro_f1':sum(v[1]for v in vals)/90,'macro_precision':sum(v[2]for v in vals)/90,'macro_recall':sum(v[3]for v in vals)/90,'exact_accuracy':sum(v[4]for v in vals)/90,'records_sha256':sha(d/'records.json'),'semantic_evidence':None,'status':'PASS'}
print(json.dumps({'manifest_sha256':sha(P/'manifest.json'),'validation_only':True,'cells':out,'provider_calls':0}))
