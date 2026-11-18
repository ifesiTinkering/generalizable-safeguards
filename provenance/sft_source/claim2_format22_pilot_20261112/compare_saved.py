"""Offline comparison of two complete, same-row v9 validation packets. No clients."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path

def compare(a,b):
 if not a['complete'] or not b['complete']:raise ValueError('Incomplete provider/judge packet; do not substitute zero')
 if a['populations']['all']!=b['populations']['all']:raise ValueError('Expected same reporting population')
 if set(a['tracked_metrics'])!=set(b['tracked_metrics']):raise ValueError('Metric roster mismatch')
 return {'scope':'heldout comparison, fresh raw base vs22-row answer-and-format SFT; one fixed sample perrow', 'conditions':{'fresh_base':a,'format22':b},'deltas_format22_minus_base':{k:None if a['tracked_metrics'][k] is None or b['tracked_metrics'][k] is None else b['tracked_metrics'][k]-a['tracked_metrics'][k] for k in a['tracked_metrics']},'format_validity':{k:v['populations']['parseable']/v['populations']['all'] for k,v in [('fresh_base',a),('format22',b)]},'interpretation':'All-row semantic scores include malformed zero; valid-only metrics use the explicitly stated parseable population. Missing semantic judgments are incomplete, never inferred zeros. No significance claim from one evaluation draw.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--code',required=True);p.add_argument('--rows',required=True);p.add_argument('--base',required=True);p.add_argument('--format22',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 sys.path.insert(0,a.code);adapter=importlib.import_module('pilot_metric_adapter');rows=json.loads(Path(a.rows).read_text());ids={r['row_id'] for r in rows};assert len(rows)==len(ids)==488
 manifest=json.loads(Path(a.manifest).read_text());assert set(manifest['row_ids'])==ids; assert manifest['n']==488
 expected_config=manifest['sampling_config']; assert {'thinking','temperature','max_tokens'}<=set(expected_config)
 records={k:json.loads(Path(v).read_text()) for k,v in [('fresh_base',a.base),('format22',a.format22)]}
 for rs in records.values():assert len(rs)==488 and {r['row_id'] for r in rs}==ids
 left={r['row_id']:r for r in records['fresh_base']};right={r['row_id']:r for r in records['format22']}
 for rid in ids:
  if left[rid].get('sampling_config')!=expected_config or right[rid].get('sampling_config')!=expected_config:raise ValueError('Sampling configuration mismatch')
  for key in ('prompt_sha256','sampling_seed'):
   if key not in left[rid] or key not in right[rid] or left[rid][key]!=right[rid][key]:raise ValueError('Unmatched prompt/seed: '+rid+' '+key)
 result=compare(*(adapter.metric_values(records[k],rows) for k in ['fresh_base','format22']))
 result['subsets']={}
 for split in sorted({r['split'] for r in rows}):
  subset=[r for r in rows if r['split']==split];sid={r['row_id'] for r in subset}
  result['subsets'][split]=compare(*(adapter.metric_values([r for r in records[k] if r['row_id'] in sid],subset) for k in ['fresh_base','format22']))
 for domain in sorted({r['domain'] for r in rows}):
  subset=[r for r in rows if r['domain']==domain];sid={r['row_id'] for r in subset}
  result['subsets']['domain:'+domain]=compare(*(adapter.metric_values([r for r in records[k] if r['row_id'] in sid],subset) for k in ['fresh_base','format22']))
 result['source_sha256']={str(v):hashlib.sha256(Path(v).read_bytes()).hexdigest() for v in [a.rows,a.base,a.format22,Path(__file__).with_name('pilot_metric_adapter.py'),a.manifest]}
 out=Path(a.output);out.write_text(json.dumps(result,indent=2)+'\n')
 table=['# Fresh-base versus 22-row SFT heldout comparison','', 'Same488 heldout rows; one sample per row. Format validity is separate from semantic quality.','', '|Metric|Fresh base|Format22|Change|','|---|---:|---:|---:|']
 fmt=lambda x:'undefined' if x is None else f'{x:.6f}'
 for k in sorted(result['deltas_format22_minus_base']):table.append('|'+ '|'.join([k,fmt(result['conditions']['fresh_base']['tracked_metrics'][k]),fmt(result['conditions']['format22']['tracked_metrics'][k]),fmt(result['deltas_format22_minus_base'][k])])+'|')
 out.with_suffix('.md').write_text('\n'.join(table)+'\n')
if __name__=='__main__':main()
