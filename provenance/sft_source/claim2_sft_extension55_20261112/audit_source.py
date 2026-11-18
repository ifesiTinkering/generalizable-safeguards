import json,sys,re,hashlib
from pathlib import Path
P=Path('/workspace/claim2-evidence-ablations-20261111/claim2_sft_extension55_20261112');sys.path.insert(0,str(P));import extend
read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();m=read(P/'manifest.json');assert sha(P/'manifest.json')=='753a5bb28ed28d2a7d49cbfad3795f4ba1007fdb5cfb25494aecc0c56c458b2d';assert all(sha(f)==h for f,h in m['source_hashes'].items());extend.paused();assert extend.lineage()==m['parent'];assert not(extend.OUT/'launch.json').exists()
rows=read(P/'new23_rows.json');allrows=read(P/'cumulative55_rows.json');prior=read(extend.PILOT/'rows32.json');assert len({r['row_id']for r in rows})==23 and len({r['row_id']for r in allrows})==55;assert not(set(r['row_id']for r in prior)&set(r['row_id']for r in rows));assert sum(bool(r['ground_truth_rule_ids'])for r in allrows)==50;assert len(extend.batches(rows))==18
for rid,d in read(P/'prepared_datums.json').items():
 a=d['audit'];t=a['target'];spans=a['masked_spans'];assert {s['field']for s in spans}<={'turn_id','quote','justification','analysis','confidence'};assert a['response_weights'][-1]==1
 required=[]
 for regex in [r'<rule id="([^"]*)"',r'speaker="([^"]*)"',r'</(?:answer|rule|quote|justification|evidence|analysis|confidence)>']:
  for match in re.finditer(regex,t):required.append(match.span(1)if match.lastindex else match.span())
 for begin,end in required:
  overlapping=[w for (x,y),w in zip(a['offsets'],a['response_weights'])if x<end and y>begin];assert overlapping and all(w==1 for w in overlapping),(rid,t[begin:end])
 for s in spans:
  overlapping=[w for (x,y),w in zip(a['offsets'],a['response_weights'])if x<s['end']and y>s['start']];assert all(w==0 for w in overlapping)
 assert not any(d['weights'][:a['response_start']-1]);assert len(d['weights'])==len(d['target_tokens'])==len(d['input_tokens'])
print(json.dumps({'decision':'GO_ONCE_ONLY','manifest_sha256':sha(P/'manifest.json'),'source_hashes_verified':True,'actual_token_mask_cases':23,'all_closures_and_evidenceIDs_speaker_supervised':True,'semantic_placeholders_masked':True,'unique_new_positive':20,'unique_new_negative':3,'cumulative_positive':50,'cumulative_negative':5,'optimizer_restore':'create_training_client_from_state_with_optimizer','parent_checkpoint':m['parent']['checkpoint'],'new_updates':18,'total_updates':45,'new_exposures':69,'bound':m['full_upper_usd'],'old48_paused':True,'extension_not_dispatched':True,'provider_calls':0}))
