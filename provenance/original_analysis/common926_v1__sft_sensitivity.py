"""One shared SFT, same original384 test contract on uniform retained926."""
from pathlib import Path
import sys,json,gzip,itertools,collections,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
D=Path(__file__).resolve().parent;O=D.parent;sys.path.insert(0,str(O))
import analyze_sft_contrasts as a
from cache_qwen_sft930 import SAMPLER,BASE
from weighted_panel_metrics import array,calculate
from count_packet_fixed24 import summarize_count_records
from analyze_full_panel_arrays import load,sha
def dump(p,x):
 with p.open('xb')as f:f.write(gzip.compress(json.dumps(x,sort_keys=True,allow_nan=False).encode(),mtime=0))
def baseline():
 p=D/'QWEN_SFT_ARRAYS.json.gz'
 if p.exists():return load(p)
 plan=load(D/'PLAN.json');rows={r['row_id']:r for r in plan['retained_rows']};cells=[];bindings={str(D/'PLAN.json'):sha(D/'PLAN.json')}
 for mode,folder,name in [('OFF','off_after_grade_v1','row_proofs.json.gz'),('TWO_STAGE_ON','on_join_v1/results','row_audit.json.gz')]:
  pp=BASE/folder/name;proof=load(pp);bindings[str(pp)]=sha(pp);proof=proof['row_proofs']if isinstance(proof,dict)else proof;groups=collections.defaultdict(list)
  for r in proof:
   rid=r['row_id']
   if rid not in rows:continue
   m=r['normalized_metrics'];v=bool(m['valid']);g=len(rows[rid]['gold_rule_ids']);assert m['_gold_count']==g;t=int(m['TP'])if v else 0;fp=int(m['FP'])if v else 0;fn=int(m['FN'])if v else g;assert t+fn==g
   f=2*t/(2*t+fp+fn)if t+fp+fn else 1.;j=t/(t+fp+fn)if t+fp+fn else 1.;exact=float(fp+fn==0);assert abs(m['quality_f1']-(f if v else 0.))<1e-12
   groups[rows[rid]['source']+':'+rows[rid]['domain']].append({'row_id':rid,'draw':r['replicate'],'valid':v,'confidence':m['confidence']if v else None,'f1':f if v else 0.,'iou':j if v else 0.,'exact':exact if v else 0.,'gold_rules':rows[rid]['gold_rule_ids'],'predicted_count':t+fp,'tp':t,'fp':fp,'fn':fn,'evidence_fraction':m.get('semantic_fraction'),'evidence_all':m.get('semantic_all'),'evidence_coverage':m.get('coverage'),'semantic_unavailability':r.get('semantic_unavailability')})
  for domain,rs in groups.items():
   rs.sort(key=lambda r:(r['row_id'],r['draw']));wanted=sorted((rid,d)for rid,r in rows.items()if r['source']+':'+r['domain']==domain for d in range(3));assert[(r['row_id'],r['draw'])for r in rs]==wanted;v=array(rs)
   cells.append({'identity':{'target':'SFT','arm':None,'seed':None,'mode':mode,'cell':domain,'sampler':SAMPLER},'values':v.tolist(),'metrics':calculate(v),'fixed24':summarize_count_records(rs),'row_ids':[r['row_id']for r in rs],'draws':[r['draw']for r in rs],'conversation_ids':[rows[r['row_id']]['conversation_hash']for r in rs]})
 out={'cells':cells,'provider_calls':0,'source_bindings':bindings,'source_sha256':sha(__file__),'known_records':sum(len(c['values'])for c in cells)};assert out['known_records']==5556;dump(p,out);return out
def init():
 q=load(D/'QWEN_ARRAYS.json.gz');g=load(D/'GPT_ARRAYS.json.gz');b=baseline()
 for family,data,base in [('Qwen',q,b),('GPT',g,g)]:
  a.DATA[family]={'rl':{tuple(c['identity'][k]for k in('target','arm','seed','mode','cell')):c for c in data['cells']if c['identity']['target']in('binary','iou')},'base':{(c['identity']['mode'],c['identity']['cell']):c for c in base['cells']if c['identity']['target']=='SFT'},'full':{tuple(c['identity'][k]for k in('target','arm','seed','mode','cell')):c['summary']['metrics']for c in data['fixed24_cells']}}
 a.DATA['Qwen']['base_full']={(c['identity']['mode'],c['identity']['cell']):c['fixed24']['metrics']for c in b['cells']}
 a.DATA['GPT']['base_full']={(c['identity']['mode'],c['identity']['cell']):c['summary']['metrics']for c in g['fixed24_cells']if c['identity']['target']=='SFT'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--points-only',action='store_true');p.add_argument('--workers',type=int,default=3);args=p.parse_args();baseline();specs=list(itertools.product(['Qwen','GPT'],['binary','iou'],['safepyramid','dynabench'],['OFF','TWO_STAGE_ON'],['ID','OOD'],['A','B']))
 if args.points_only:init();results=[a.prepare(s)[0]for s in specs]
 else:
  results=[]
  with ProcessPoolExecutor(max_workers=args.workers,initializer=init)as pool:
   futures={pool.submit(a.compute,s):s for s in specs}
   for f in as_completed(futures):
    r=f.result();results.append(r)
    if r['complete']:print(json.dumps({'finished':futures[f]}),flush=True)
  a.adjust(results)
 out={'schema':'common926-single-SFT-sensitivity-v1','provider_calls':0,'results':results,'family_size':384,'plan_sha256':sha(D/'PLAN.json'),'original_SFT_plan_sha256':sha(O/'SFT_COMPARISON_ANALYSIS_PLAN_V1.json'),'source_sha256':sha(__file__),'inference_source_sha256':sha(O/'analyze_sft_contrasts.py'),'source_bindings':{str(D/n):sha(D/n)for n in('QWEN_ARRAYS.json.gz','GPT_ARRAYS.json.gz','QWEN_SFT_ARRAYS.json.gz')},'limits':'Uniform common926 sensitivity, one shared SFT paired with fixed RL checkpoints; original384 results unchanged; no baseline-as-seed pseudoreplication.'};p=D/('SFT_POINTS.json.gz'if args.points_only else'SFT_INFERENCE.json.gz');dump(p,out);print(json.dumps({'path':str(p),'sha256':sha(p)}))
if __name__=='__main__':main()
