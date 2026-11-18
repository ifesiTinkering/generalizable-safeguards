"""Format-only optimizer continuation; no correctness/evidence payload CE."""
import os,sys,json,time,random,threading,hashlib,fcntl
from pathlib import Path
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
PILOT=HERE.parent/'claim2_format22_pilot_20261112';CODE=HERE.parent/'runtime_v9_tinker_template'
sys.path[1:1]=[str(PILOT),str(CODE)]
from protocol import read,save,sha
from target_builder import build_record
from objective_prompts import messages
from thinking_adapter import renderer,decode_with_audit
from reward_pipeline import parse,score_all
from cost_plan import training_upper_bound,sampling_upper_bound
from data.data_recipe import final_sampling_seed,training_sampling_seed
from heldout_metric_adapter import metric_values
ROOT=Path('/workspace/safepyramid-runs/claim2_evidence_ablations_20261111/partial')
PRIOR=ROOT/'pilot_sft_extension55_20261112';OUT=ROOT/'pilot_format_only55_20261112'
def paused():
 for pid in(973394,973395):
  assert '\nState:\tT'in Path(f'/proc/{pid}/status').read_text(),'RL must remain paused'
def batches(rows,epoch):
 rr=sorted(rows,key=lambda x:x['row_id']);random.Random(epoch).shuffle(rr);return[rr[i:i+4]for i in range(0,len(rr),4)]
def parent():
 state=read(PRIOR/'state.json');assert state=={'new_updates':18,'total_updates':45,'pending':None}
 return read(PRIOR/'final_checkpoint.json')
def prepare():
 rows=read(HERE/'training_rows.json');vr=read(HERE/'validation90_rows.json');test=read(PILOT/'validation_rows.json')
 assert len(rows)==55 and sum(bool(x['ground_truth_rule_ids'])for x in rows)==50
 assert len(vr)==90 and len({x['row_id']for x in vr})==90
 assert not({x['conversation_sha256']for x in rows}&{x['conversation_sha256']for x in vr+test})
 assert not({x['conversation_sha256']for x in vr}&{x['conversation_sha256']for x in test})
 r=renderer(False);ds={x['row_id']:build_record(r,messages(x,'partial','A'),x)for x in rows}
 for row in rows:parse(ds[row['row_id']]['audit']['target'],[str(x['rule_id'])for x in row['all_policy_rules']])
 vp={x['row_id']:r.build_generation_prompt(messages(x,'partial','A'),prefill='<answer>\n').to_ints()for x in vr}
 save(HERE/'prepared_datums.json',ds);save(HERE/'validation_prompts.json',vp)
 train=training_upper_bound([len(ds[x['row_id']]['input_tokens'])+1 for epoch in range(10)for b in batches(rows,epoch)for x in b])
 val=sum((sampling_upper_bound(len(p),23000)for p in vp.values()),Decimal(0))*6
 fp=read(PILOT/'validation_prompts.json');final=sum((sampling_upper_bound(len(p),10000 if mode=='ON' else 23000)for mode,ps in fp.items()for p in ps.values()),Decimal(0))
 bound=train+val+final+Decimal('1.4')
 sources={str(p):sha(p)for p in HERE.glob('*.py')}
 for n in ['training_rows.json','validation90_rows.json','prepared_datums.json','validation_prompts.json']:sources[str(HERE/n)]=sha(HERE/n)
 m={'source_hashes':sources,'parent':parent(),'parent_updates':45,'parent_source_manifest_sha256':sha(HERE.parent/'claim2_sft_extension55_20261112/manifest.json'),'original_pilot_manifest_sha256':sha(PILOT/'manifest.json'),'restore_optimizer':True,'max_epochs':10,'max_updates':140,'validation_every_epochs':2,'validation_n':90,'validity_threshold_count':86,'selection':'First scheduled validation with >=86/90 valid, otherwise final10epoch checkpoint. F1/P not used. Baseline alreadypassing => no updates.','learning_rate':1e-5,'adam':{'beta1':.9,'beta2':.95,'eps':1e-8},'train_upper_usd':str(train),'validation_upper_usd':str(val),'final_eval_upper_usd':str(final),'storage_allowance_usd':'1.4','full_upper_usd':str(bound),'objective':'Only syntax/punctuation/closing/nesting/allowed speaker enum CE. Answer IDs, evidence IDs, turns, quotes, justification, analysis, confidence masked. Source-derived contexts not asserted gold evidence.','validation':'OFF23000 exact frozen90 deterministic parser; no judge. Correctnessdiagnostic only.','final_evaluation':'Once selected checkpoint on exact488 OFF23000/ON10000TOTAL. No finalpanel selection.'}
 save(HERE/'manifest.json',m);print(json.dumps(m|{'source_hashes':None},indent=2))
def execute():
 paused();m=read(HERE/'manifest.json');a=read(HERE/'authorization.json');assert a.get('approved')is True and a['manifest_sha256']==sha(HERE/'manifest.json')
 for p,h in m['source_hashes'].items():assert sha(p)==h
 assert sha(HERE.parent/'claim2_sft_extension55_20261112/manifest.json')==m['parent_source_manifest_sha256']
 assert sha(PILOT/'manifest.json')==m['original_pilot_manifest_sha256']and parent()==m['parent']
 OUT.mkdir(exist_ok=True)
 with(OUT/'lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with(OUT/'launch.json').open('x')as f:json.dump({'pid':os.getpid(),'unix':time.time(),'manifest_sha256':sha(HERE/'manifest.json')},f)
  for line in Path('/workspace/.safepyramid.env').read_text().splitlines():
   if '='in line and not line.lstrip().startswith('#'):
    k,v=line.strip().removeprefix('export ').split('=',1)
    if k in('TINKER_API_KEY','OPENAI_API_KEY'):os.environ.setdefault(k,v.strip().strip('\"\''))
  from evidence_budget import Ledger
  from cohort_budget import CohortMeter
  from astra_judge import AstraJudge,JudgeUnavailable
  from live_backend import disable_retries
  from sft_backend import SFTBackend
  from tinker.lib.retry_handler import RetryConfig
  import tinker
  ledger=Ledger(ROOT/'budget.sqlite',cap_usd=1000);meter=CohortMeter(ledger,'evidence:formatonly55',Decimal(m['full_upper_usd']),OUT/'cohort.json')
  meter.reserve('evidence:formatonly55:storage',Decimal('1.4'),{'kind':'modeled_storage_allowance'});meter.dispatch('evidence:formatonly55:storage');meter.settle('evidence:formatonly55:storage',Decimal('1.4'))
  b=SFTBackend.__new__(SFTBackend);b.config={'model':'Qwen/Qwen3.5-9B','initial_checkpoint':m['parent']['checkpoint'],'learning_rate':1e-5,'adam':m['adam'],'storage_reserve_confirmed':True,'storage_max_pairs_per_job':7};b.outdir=OUT;b.journal=OUT/'transport_journal';b.journal.mkdir();b.meter=meter;b.sdk=tinker;b.retry_class=RetryConfig;b.service=tinker.ServiceClient();disable_retries(b.service);b.client=None;b.sampler=None;b.sampling_path=None
  try:
   def restore():
    b.client=b.service.create_training_client_from_state_with_optimizer(m['parent']['checkpoint']);disable_retries(b.client);b.set_sampling_path(m['parent']['sampling_path']);return{'restore_optimizer':True,'checkpoint':m['parent']['checkpoint']}
   b._execute('evidence:formatonly55:restore',{'kind':'restore_optimizer','checkpoint':m['parent']['checkpoint']},Decimal(0),restore,allow_cached=False)
   rows=read(HERE/'training_rows.json');vr=read(HERE/'validation90_rows.json');vp=read(HERE/'validation_prompts.json');ds=read(HERE/'prepared_datums.json');decode_lock=threading.Lock()
   def evaluate(rows,prompts,mode,dest,prefix,judge=None):
    r=renderer(mode=='ON');b.config['stop_sequences']=r.get_stop_sequences();limit=10000 if mode=='ON' else 23000
    def one(row):
     seed=training_sampling_seed(0,'validation',row['row_id'],0) if prefix.startswith('evidence:formatonly55:validation:') else final_sampling_seed(row['split'],row['row_id']);seq=b.sample(prompts[row['row_id']],seed,limit,prefix+':'+row['row_id'])
     with decode_lock:de=decode_with_audit(r,seq['tokens'],thinking=mode=='ON')
     raw={'row_id':row['row_id'],'text':de['text'],'clean_termination':de['clean'],'split':row.get('split','validation'),'mode':mode,'sampling_seed':seed,'tokens':seq['tokens'],'token_logprobs':seq['logprobs'],'decoder_audit':de,'prompt_sha256':hashlib.sha256(json.dumps(prompts[row['row_id']]).encode()).hexdigest(),'sampling_config':{'thinking':mode=='ON','temperature':1.0,'max_tokens':limit}}
     key=hashlib.sha256(row['row_id'].encode()).hexdigest();save(dest/'raw'/(key+'.json'),raw)
     try:scores=score_all(row,raw['text'],'partial',judge,raw['clean_termination'])
     except JudgeUnavailable as exc:scores=dict(score_all(row,raw['text'],'partial',None,raw['clean_termination']),judge_unavailable=True,reason=str(exc),correctness_derived_offline=True)
     rec=raw|scores;save(dest/'scored'/(key+'.json'),rec);return rec
    with ThreadPoolExecutor(max_workers=32)as pool:records=list(pool.map(one,rows))
    save(dest/'records.json',records);return records
   def validation(epoch):
    dest=OUT/'validation'/f'epoch{epoch:02d}';rr=evaluate(vr,vp,'OFF',dest,'evidence:formatonly55:validation:'+str(epoch));n=sum(bool(x.get('valid'))for x in rr);receipt={'n':90,'valid':n,'pass':n>=86,'epoch':epoch,'records_sha256':sha(dest/'records.json'),'judge_calls':0,'semantic_evidence_evaluated':False,'selection_metric':'full_contract_validity'};save(dest/'complete.json',receipt);return receipt
   base=validation(0);selected=m['parent'];done=0;passed=base['pass'];save(OUT/'state.json',{'updates':0,'total_updates':45,'pending':None,'phase':'format_only','format_pass':passed})
   if not passed:
    for epoch in range(1,11):
     for batch in batches(rows,epoch-1):
      paused();save(OUT/'state.json',{'updates':done,'total_updates':45+done,'pending':done+1,'epoch':epoch});result=b.supervised_update([ds[x['row_id']]for x in batch],'evidence:formatonly55:update:'+str(done+1));done+=1;save(OUT/'updates'/f'{done:03d}.json',result);save(OUT/'state.json',{'updates':done,'total_updates':45+done,'pending':None,'epoch':epoch})
     if epoch%2==0:
      pair=b.checkpoint(f'formatonly-epoch{epoch:02d}');b.retain(pair,604800);save(OUT/'checkpoints'/f'epoch{epoch:02d}.json',pair);v=validation(epoch);selected=pair;passed=v['pass'];save(OUT/'selection_progress.json',v|{'checkpoint':pair,'updates':done})
      if passed:break
   save(OUT/'selected_checkpoint.json',{'checkpoint_pair':selected,'updates':done,'format_gate_passed':passed,'selection':'first>=86/90elsefinal','no_F1_selection':True})
   # The sampler is the immutable selected last saved checkpoint (or parent if baseline passes).
   judge=AstraJudge(ROOT)
   with judge.ledger.tx()as db:
    cap=db.execute('select max_calls from grant_meta').fetchone()[0];used=db.execute("select count(*)from grant_calls where status!='cancelled'").fetchone()[0]
   if used+976>cap:raise RuntimeError('Insufficient final semantic grant capacity; never replay training')
   test=read(PILOT/'validation_rows.json');prompts=read(PILOT/'validation_prompts.json')
   for mode in('OFF','ON'):
    dest=OUT/'evaluations'/mode;rr=evaluate(test,prompts[mode],mode,dest,'evidence:formatonly55:final:'+mode,judge);packet=metric_values(rr,test);save(dest/'metrics.json',packet);receipt={'sampling_complete':True,'correctness_complete':True,'semantic_complete':packet['complete'],'n':488,'records_sha256':sha(dest/'records.json')};save(dest/'sampling_complete.json',receipt)
    if packet['complete']:save(dest/'complete.json',receipt|{'complete':True})
   actual=meter.close();save(OUT/'completed.json',{'updates':done,'total_updates':45+done,'format_gate_passed':passed,'selected':selected,'cash_actual_or_allowance_usd':str(actual)})
  except BaseException as exc:save(OUT/'failure.json',{'type':type(exc).__name__,'message':str(exc),'no_retry':True});raise
  finally:b.close(status='success'if(OUT/'completed.json').exists()else'errored')
if __name__=='__main__':{'prepare':prepare,'execute':execute}[sys.argv[1]]()
