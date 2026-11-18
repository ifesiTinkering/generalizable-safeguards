import sys,json,hashlib,re
from pathlib import Path
P=Path('/workspace/claim2-evidence-ablations-20261111/claim2_format_only55_20261112');sys.path.insert(0,str(P));import format_only as f
from thinking_adapter import renderer,build_generation_prompt
from data.data_recipe import training_sampling_seed
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();m=read(P/'manifest.json');assert all(sha(p)==h for p,h in m['source_hashes'].items());f.paused();assert not(f.OUT/'launch.json').exists();assert f.parent()==m['parent']
for mf in [f.PILOT/'manifest.json',P.parent/'claim2_sft_extension55_20261112/manifest.json']:
 pm=read(mf);assert all(sha(p)==h for p,h in pm['source_hashes'].items())
rows=read(P/'training_rows.json');vr=read(P/'validation90_rows.json');held=read(f.PILOT/'validation_rows.json');assert len(rows)==55 and len({r['row_id']for r in rows})==55 and sum(bool(r['ground_truth_rule_ids'])for r in rows)==50;assert len(vr)==90 and len(held)==488
assert not({r['conversation_sha256']for r in rows}&{r['conversation_sha256']for r in vr+held});assert not({r['conversation_sha256']for r in vr}&{r['conversation_sha256']for r in held})
ds=read(P/'prepared_datums.json');assert len(ds)==55
for d in ds.values():
 a=d['audit'];t=a['target'];spans=a['masked_spans'];assert not a['answer_ids_supervised']and not a['evidence_rule_ids_supervised'];assert a['speaker_enum_supervised'];assert a['response_weights'][-1]==1
 for s in spans:
  ww=[w for (x,y),w in zip(a['offsets'],a['response_weights'])if x<s['end']and y>s['start']];assert ww and all(w==0 for w in ww)
 for regex in [r'</(?:answer|rule|quote|justification|evidence|analysis|confidence)>',r'speaker="assistant"']:
  for mt in re.finditer(regex,t):
   ww=[w for (x,y),w in zip(a['offsets'],a['response_weights'])if x<mt.end()and y>mt.start()];assert ww and all(w==1 for w in ww)
 assert not any(d['weights'][:a['response_start']-1]);assert len(d['weights'])==len(d['input_tokens'])==len(d['target_tokens']);assert a['full_tokens']<=32768
r=renderer(False);vp=read(P/'validation_prompts.json')
for row in vr:
 assert vp[row['row_id']]==build_generation_prompt(r,f.messages(row,'partial','A'),thinking=False).to_ints();assert len(vp[row['row_id']])+23000<=32768;training_sampling_seed(0,'validation',row['row_id'],0)
assert len(f.batches(rows,0))==14 and len(f.batches(rows,0)[-1])==3;assert m['max_updates']==140 and m['validity_threshold_count']==86
print(json.dumps({'decision':'GO_ONCE_ONLY','manifest_sha256':sha(P/'manifest.json'),'source_hashes_verified':True,'actual_token_mask_cases':55,'all_ID_payloads_and_semantics_masked':True,'speaker_and_closures_supervised':True,'validation_rows':90,'validation_source_conversations':len({x['conversation_sha256']for x in vr}),'validation_prompt_matches_current_OFF':True,'validation_seed_works':True,'train_validation_final_sources_disjoint':True,'restore_optimizer_parent':m['parent'],'max_updates':140,'stop_count':86,'bound':m['full_upper_usd'],'old48_paused':True,'not_dispatched':True,'provider_calls':0}))
