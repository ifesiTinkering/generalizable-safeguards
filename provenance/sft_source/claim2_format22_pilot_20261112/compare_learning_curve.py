"""Offline four-condition heldout comparison; no provider clients or selection."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
CONDITIONS=('fresh_base','format12','format22','format32')
def verify(rows,records,manifest):
 ids={r['row_id'] for r in rows}
 if len(rows)!=488 or len(ids)!=488 or set(manifest['row_ids'])!=ids or manifest['n']!=488:raise ValueError('Exact488 manifest population required')
 config=manifest['sampling_config']
 if not {'thinking','temperature','max_tokens'}<=set(config):raise ValueError('Missing sampling settings')
 if set(records)!=set(CONDITIONS):raise ValueError('Allfour conditions required')
 base={r['row_id']:r for r in records['fresh_base']}
 for name,rs in records.items():
  if len(rs)!=488 or len({r['row_id'] for r in rs})!=488 or {r['row_id'] for r in rs}!=ids:raise ValueError('Condition population mismatch')
  for r in rs:
   if r.get('sampling_config')!=config:raise ValueError('Sampling config mismatch')
   for k in ('prompt_sha256','sampling_seed'):
    if k not in r or k not in base[r['row_id']] or r[k]!=base[r['row_id']][k]:raise ValueError('Prompt/seed mismatch')
 return True

def compute(rows,records,metric_values):
 packets={k:metric_values(rs,rows) for k,rs in records.items()}
 if any(not p['complete'] for p in packets.values()):raise ValueError('Incompleteprovider/judge packet')
 names=set(packets['fresh_base']['tracked_metrics'])
 if any(set(p['tracked_metrics'])!=names for p in packets.values()):raise ValueError('Metric roster mismatch')
 base=packets['fresh_base']['tracked_metrics']
 return {'conditions':packets,'format_validity':{k:p['populations']['parseable']/p['populations']['all'] for k,p in packets.items()},'deltas_vs_fresh_base':{k:{m:None if v is None or base[m] is None else v-base[m] for m,v in p['tracked_metrics'].items()} for k,p in packets.items() if k!='fresh_base'}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--code',required=True);p.add_argument('--rows',required=True);p.add_argument('--manifest',required=True);p.add_argument('--records-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 sys.path.insert(0,a.code);sys.path.insert(0,str(Path(__file__).parent));adapter=importlib.import_module('pilot_metric_adapter')
 rows=json.loads(Path(a.rows).read_text());manifest=json.loads(Path(a.manifest).read_text());paths={k:Path(a.records_root)/k/'records.json' for k in CONDITIONS};records={k:json.loads(v.read_text()) for k,v in paths.items()};verify(rows,records,manifest)
 result=compute(rows,records,adapter.metric_values);result['subsets']={}
 for field in ('split','domain'):
  for value in sorted({r[field] for r in rows}):
   sub=[r for r in rows if r[field]==value];ids={r['row_id'] for r in sub};result['subsets'][field+':'+value]=compute(sub,{k:[r for r in rs if r['row_id'] in ids] for k,rs in records.items()},adapter.metric_values)
 result['scope']='OnecontinuousSFTtrajectory checkpoints9/18/27; sharedbaseevaluatedonce; same488heldoutpanel/settings; counts10/20/30positiveeachsame2negative. Panelisnowselectionexposed ifusedtochooseRLinitialization.'
 result['source_sha256']={str(v):hashlib.sha256(Path(v).read_bytes()).hexdigest() for v in [a.rows,a.manifest,*paths.values(),Path(__file__).with_name('pilot_metric_adapter.py')]}
 out=Path(a.output);out.write_text(json.dumps(result,indent=2)+'\n');lines=['# Freshbase and12/22/32-row SFT heldout comparison','','Format validity is reported separately. No significance claim from one draw perrow.','','|Metric|Freshbase|10positive+2negative|20positive+2negative|30positive+2negative|','|---|---:|---:|---:|---:|']
 fmt=lambda x:'undefined' if x is None else f'{x:.6f}'
 lines.append('|Formatvalidity|'+'|'.join(fmt(result['format_validity'][k]) for k in CONDITIONS)+'|')
 for m in sorted(result['conditions']['fresh_base']['tracked_metrics']):lines.append('|'+m+'|'+'|'.join(fmt(result['conditions'][k]['tracked_metrics'][m]) for k in CONDITIONS)+'|')
 out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
