import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
files=[R/'completed_cells_full_audit.json',R/'incomplete_semantic_student_audit.json'];a,b=[json.loads(p.read_text()) for p in files];cells={}
for name in ['fresh_base','format12','format22','format32']:
 if name in ['fresh_base','format12']:
  c=a['cells']['OFF/'+name];metrics=c['metrics'];parseable=c['parseable'];status='All488 student and semantic results verified'
 else:
  c=b['cells'][name];metrics=dict(c['nonsemantic_metrics'],semantic_aggregate_evidence=None,semantic_per_rule_evidence=None);parseable=c['populations']['parseable'];status='All488 student results verified; semantic evidence incomplete (one policy-rejected judge row)'
 cells[name]={'n':488,'parseable':parseable,'valid_rate':parseable/488,'metrics':metrics,'status':status}
out={'scope':'OFF23k,488 identical heldout rows, one continuous SFT trajectory; freshbase shared','conditions':cells,'audit_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'warning':'For20/30 positives, uses corrected saved-student-text recomputation; original incomplete packets incorrectly counted one valid judge-rejected row as malformed. Semantic evidence unavailable for these two cells; no zeros substituted.'}
(R/'off_corrected_summary.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# OFF comparison: fresh base and continuous SFT milestones','','All488 student outputs verified. Evidence scores for20/30-positive milestones remain unavailable because one semantic-judge row was policy rejected.','','|Condition|Valid|SetF1|Precision|Recall|Exact|','|---|---:|---:|---:|---:|---:|']
labels={'fresh_base':'Fresh base','format12':'10positive+2negative','format22':'20positive+2negative','format32':'30positive+2negative'}
for name,c in cells.items():m=c['metrics'];lines.append('|'+labels[name]+'|'+'|'.join(f'{100*x:.2f}%' for x in [c['valid_rate'],m['macro_f1'],m['macro_precision'],m['macro_recall'],m['exact_set_accuracy']])+'|')
lines+=['','Quality metrics include malformed outputs as zero; confidence metrics use parseable outputs. The heldoutpanel is selection-exposed if used to choose initialization. No inference was performed by this extraction.','','|Metric|Fresh base|10positive|20positive|30positive|','|---|---:|---:|---:|---:|']
for metric in cells['fresh_base']['metrics']:
 lines.append('|'+metric+'|'+'|'.join('N/A' if cells[k]['metrics'][metric] is None else f"{cells[k]['metrics'][metric]:.6f}" for k in cells)+'|')
(R/'OFF_CORRECTED_TABLES.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:11]))
