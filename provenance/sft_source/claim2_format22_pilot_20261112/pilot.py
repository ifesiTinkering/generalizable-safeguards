"""Once-only22-row format SFT + raw-base/final shared90 validation; no RL."""
import sys,json,hashlib,os,time,random,threading,fcntl
from pathlib import Path
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
CODE=Path('/workspace/claim2-evidence-ablations-20261111/runtime_v9_tinker_template')
sys.path.insert(1,str(CODE))
from protocol import read,save,sha
from target_builder import build_record
from objective_prompts import messages
from data.data_recipe import load_rows,final_sampling_seed
from thinking_adapter import renderer as mode_renderer,build_generation_prompt,decode_with_audit
from cost_plan import sampling_upper_bound,training_upper_bound
OUT=Path('/workspace/safepyramid-runs/claim2_evidence_ablations_20261111/partial/pilot_sft_learningcurve_20261112')
BUDGET=OUT.parent

def renderer():
 from transformers import AutoTokenizer
 from tinker_cookbook import renderers
 return renderers.get_renderer('qwen3_5_disable_thinking',AutoTokenizer.from_pretrained('/workspace/tokenizer-qwen35',local_files_only=True))
def paused():
 for pid in(973394,973395):
  text=Path(f'/proc/{pid}/status').read_text()
  if '\nState:\tT' not in text:raise RuntimeError('Original supervisors must remain paused')
def ordered(rows):
 batches=[]
 for epoch in range(3):
  rr=sorted(rows,key=lambda r:r['row_id']);random.Random(epoch).shuffle(rr);batches.extend([rr[i:i+4]for i in range(0,len(rr),4)])
 return batches

def chunks(subsets):
 first=subsets['12'];ids12={r['row_id']for r in first};ids22={r['row_id']for r in subsets['22']}
 return {'12':first,'22':[r for r in subsets['22']if r['row_id']not in ids12],'32':[r for r in subsets['32']if r['row_id']not in ids22]}

def prepare():
 r=renderer();subsets={str(n):read(HERE/('rows'+str(n)+'.json'))for n in(12,22,32)}
 for n,rr in subsets.items():assert len(rr)==int(n) and sum(bool(x['ground_truth_rule_ids'])for x in rr)==int(n)-2
 rows=subsets['32'];phases=chunks(subsets)
 panel=Path('/workspace/claim2-expanded-ood200-inventory-20261110/panel_l0');scope=read(panel/'analysis_scope_l0_v1.json');assert sha(panel/'full_rows.json')==scope['raw_full_rows_sha256']
 keys={(x['split'],x['row_id'])for x in scope['primary_keys']}
 validation=[dict(x['source_row'],split=x['split'])for x in read(panel/'full_rows.json')if(x['split'],x['row_id'])in keys]
 assert len(validation)==488 and len({x['row_id']for x in validation})==488
 assert sum(x['split']=='test_id'for x in validation)==180
 assert sum(x['split']=='test_ood'for x in validation)==308
 assert not({x['conversation_sha256']for x in rows}&{x['conversation_sha256']for x in validation})
 from reward_pipeline import parse
 datums={x['row_id']:build_record(r,messages(x,'partial','A'),x)for x in rows}
 for row in rows:
  parsed=parse(datums[row['row_id']]['audit']['target'],[str(x['rule_id'])for x in row['all_policy_rules']])
  assert set(parsed['violated_rule_ids'])==set(row['ground_truth_rule_ids'])
 renderers={mode:mode_renderer(mode=='ON')for mode in('OFF','ON')}
 prompts={mode:{x['row_id']:build_generation_prompt(renderers[mode],messages(x,'partial','A'),thinking=mode=='ON').to_ints()for x in validation}for mode in('OFF','ON')}
 assert all(len(p)+(10000 if mode=='ON'else 23000)<=32768 for mode,ps in prompts.items()for p in ps.values())
 trainbound=training_upper_bound([len(datums[x['row_id']]['input_tokens'])+1 for rr in phases.values()for batch in ordered(rr)for x in batch])
 evalbounds={mode:sum((sampling_upper_bound(len(p),10000 if mode=='ON'else 23000)for p in ps.values()),Decimal(0))*4 for mode,ps in prompts.items()}
 evalbound=sum(evalbounds.values(),Decimal(0))
 bound=trainbound+evalbound+Decimal('0.6')
 assert bound<60
 save(HERE/'prepared_datums.json',datums);save(HERE/'validation_rows.json',validation);save(HERE/'validation_prompts.json',prompts)
 sources={str(p):sha(p)for p in HERE.glob('*.py')}
 sources.update({str(p):sha(p)for p in CODE.glob('*.py')})
 for name in('rows12.json','rows22.json','rows32.json','learning_curve_manifest.json','prepared_datums.json','validation_rows.json','validation_prompts.json'):sources[str(HERE/name)]=sha(HERE/name)
 sources[str(CODE/'rates.json')]=sha(CODE/'rates.json')
 m={'source_hashes':sources,'model':'Qwen/Qwen3.5-9B','rank':64,'seed':0,'learning_rate':1e-5,'adam':{'beta1':.9,'beta2':.95,'eps':1e-8},'conditions':{n:{'cumulative_rows':int(n),'cumulative_positive':int(n)-2,'negative_seen':2,'chunk_rows':len(rr),'chunk_updates':len(ordered(rr)),'checkpoint_update':9*(i+1),'batch_sizes':[len(b)for b in ordered(rr)],'chunk_exposures':3*len(rr)}for i,(n,rr)in enumerate(phases.items())},'epochs_per_chunk':3,'total_updates':27,'total_exposures':96,'continuous_weights_and_optimizer':True,'max_tokens':23000,'evaluation_rows_per_mode_condition':488,'modes':['OFF','ON'],'evaluation_mode_upper_usd':{k:str(v)for k,v in evalbounds.items()},'train_upper_usd':str(trainbound),'evaluation_upper_usd':str(evalbound),'storage_allowance_usd':'.6','full_upper_usd':str(bound),'cash_cap_usd':'60','original48_paused':True,'source_model':'fresh_base_no_SFT_adapter','prompt':'exactv9 sharedfourtag','supervision':'CE: goldanswerIDs+structure supervised; prompt,evidencepayload+metadata,analysisbody,confidencevalues masked; tags/punctuation/Review opener/EOS supervised. Masked payload still conditions later tokens.','original_answer_JSON_stage':False,'old_tinker_update_unchanged':sha(CODE/'update.py')}
 save(HERE/'manifest.json',m);print(json.dumps(m|{'source_hashes_count':len(sources),'source_hashes':None},indent=2))

def execute():
 paused();m=read(HERE/'manifest.json');a=read(HERE/'authorization.json')
 if a.get('approved')is not True or a.get('manifest_sha256')!=sha(HERE/'manifest.json'):raise RuntimeError('Audit authorization missing')
 for p,h in m['source_hashes'].items():
  if sha(p)!=h:raise RuntimeError('Source changed:'+p)
 OUT.mkdir(exist_ok=True)
 with (OUT/'pilot.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with (OUT/'launch.json').open('x')as f:json.dump({'pid':os.getpid(),'unix':time.time(),'manifest_sha256':sha(HERE/'manifest.json')},f)
  for line in Path('/workspace/.safepyramid.env').read_text().splitlines():
   if '='in line and not line.lstrip().startswith('#'):
    k,v=line.strip().removeprefix('export ').split('=',1)
    if k in('TINKER_API_KEY','OPENAI_API_KEY'):os.environ.setdefault(k,v.strip().strip('\"\''))
  from evidence_budget import Ledger
  from cohort_budget import CohortMeter
  from astra_judge import AstraJudge,JudgeUnavailable
  from sft_backend import SFTBackend
  from live_backend import disable_retries
  from reward_pipeline import score_all
  from heldout_metric_adapter import metric_values
  from tinker.lib.retry_handler import RetryConfig
  from tinker_cookbook.renderers import get_text_content
  import tinker
  ledger=Ledger(BUDGET/'budget.sqlite',cap_usd=1000)
  judge=AstraJudge(BUDGET)
  with judge.ledger.tx()as db:
   cap=db.execute('select max_calls from grant_meta').fetchone()[0];used=db.execute("select count(*)from grant_calls where status!='cancelled'").fetchone()[0]
  if used+3904>cap:raise RuntimeError('Semantic validation grant unavailable')
  meter=CohortMeter(ledger,'evidence:sftcurve:pilot',Decimal(m['full_upper_usd']),OUT/'cohort.json')
  # Explicit bounded storage allocation, not a provider invoice.
  meter.reserve('evidence:sftcurve:storage',Decimal('.6'),{'kind':'modeled_storage_allowance'});meter.dispatch('evidence:sftcurve:storage');meter.settle('evidence:sftcurve:storage',Decimal('.6'))
  r=renderer();decode_lock=threading.Lock();sample_sem=threading.Semaphore(64);judge_sem=threading.Semaphore(32);validation=read(HERE/'validation_rows.json');prompts=read(HERE/'validation_prompts.json')
  cfg=dict(model=m['model'],learning_rate=m['learning_rate'],adam=m['adam'],stop_sequences=r.get_stop_sequences(),storage_reserve_confirmed=True,storage_max_pairs_per_job=3,initial_checkpoint='fresh_base')
  def shell(name):
   b=SFTBackend.__new__(SFTBackend);b.config=cfg;b.outdir=OUT/name;b.journal=b.outdir/'transport_journal';b.journal.mkdir(parents=True,exist_ok=True);b.meter=meter;b.sdk=tinker;b.retry_class=RetryConfig;b.service=tinker.ServiceClient();disable_retries(b.service);b.client=None;b.sampler=None;b.sampling_path=None;return b
  base=shell('fresh_base');trained=shell('continuous_training');evaluation_backends=[]
  base.sampler=base.service.create_sampling_client(base_model=m['model'],retry_config=RetryConfig(enable_retry_logic=False));disable_retries(base.sampler);base.sampling_path='raw-base:'+m['model']
  def evaluate(b,name,mode):
   rr=mode_renderer(mode=='ON');b.config=dict(cfg,stop_sequences=rr.get_stop_sequences());cap=10000 if mode=='ON'else 23000
   def one(row):
    paused();rid='evidence:sftcurve:'+name+':'+mode+':'+row['row_id'];seed=final_sampling_seed(row['split'],row['row_id'])
    with sample_sem:seq=b.sample(prompts[mode][row['row_id']],seed,cap,rid)
    with decode_lock:audit=decode_with_audit(rr,seq['tokens'],thinking=mode=='ON')
    text=audit['text']
    raw={'row_id':row['row_id'],'split':row['split'],'mode':mode,'text':text,'clean_termination':audit['clean'],'sampling_seed':seed,'tokens':seq['tokens'],'token_logprobs':seq['logprobs'],'decoder_audit':audit,'prompt_sha256':hashlib.sha256(json.dumps(prompts[mode][row['row_id']]).encode()).hexdigest(),'sampling_config':{'thinking':mode=='ON','temperature':1.0,'max_tokens':cap}}
    save(OUT/'evaluations'/mode/name/'raw'/f'{hashlib.sha256(row["row_id"].encode()).hexdigest()}.json',raw)
    try:
     with judge_sem:scores=score_all(row,text,'partial',judge,audit['clean'])
    except JudgeUnavailable as e:scores={'judge_unavailable':True,'reason':str(e)}
    record=raw|scores
    save(OUT/'evaluations'/mode/name/'scored'/f'{hashlib.sha256(row["row_id"].encode()).hexdigest()}.json',record);return record
   with ThreadPoolExecutor(max_workers=32)as pool:records=list(pool.map(one,validation))
   save(OUT/'evaluations'/mode/name/'records.json',records);packet=metric_values(records,validation);save(OUT/'evaluations'/mode/name/'metrics.json',packet)
   if not packet['complete']:raise RuntimeError('Semantic validation incomplete; no replay')
   save(OUT/'evaluations'/mode/name/'complete.json',{'complete':True,'n':488,'records_sha256':sha(OUT/'evaluations'/mode/name/'records.json')});print(json.dumps({'stage':name+'_'+mode+'_evaluation_complete'}),flush=True)
  try:
   ds=read(HERE/'prepared_datums.json')
   def baseline():
    for mode in('OFF','ON'):evaluate(base,'fresh_base',mode)
   def initialize():
    trained.client=trained.service.create_lora_training_client(base_model=m['model'],rank=64,seed=0);disable_retries(trained.client);return {'base_model':m['model'],'rank':64,'seed':0,'inherited_checkpoint':None}
   with ThreadPoolExecutor(max_workers=4)as pool:
    baseline_future=pool.submit(baseline)
    trained._execute('evidence:sftcurve:initialize',{'kind':'initialize_fresh_base','model':m['model'],'rank':64,'seed':0},Decimal(0),initialize,allow_cached=False)
    subsets={n:read(HERE/('rows'+n+'.json'))for n in('12','22','32')};pairs={};eval_futures=[];update=0
    for n,rr in chunks(subsets).items():
     condition='format'+n;dest=OUT/condition
     for batch in ordered(rr):
      paused();update+=1;records=[ds[x['row_id']]for x in batch];save(OUT/'state.json',{'updates':update-1,'pending':update,'phase':condition});result=trained.supervised_update(records,'evidence:sftcurve:update:'+str(update));save(OUT/'updates'/f'{update:03d}.json',result);save(OUT/'state.json',{'updates':update,'pending':None,'phase':condition});print(json.dumps({'stage':'sft_update','condition':condition,'update':update,'total':27}),flush=True)
     pair=trained.checkpoint(condition+'-milestone');trained.retain(pair,604800);save(dest/'final_checkpoint.json',pair);pairs[n]=pair
     # Independent immutable sampler wrapper: later optimizer/checkpoint saves
     # cannot switch the model underneath an in-flight earlier evaluation.
     evaluator=shell(condition);evaluator.set_sampling_path(pair['sampling_path']);evaluation_backends.append(evaluator)
     def eval_milestone(evaluator=evaluator,condition=condition):
      for mode in('OFF','ON'):evaluate(evaluator,condition,mode)
     eval_futures.append(pool.submit(eval_milestone))
    for future in eval_futures:future.result()
    baseline_future.result()
   actual=meter.close();save(OUT/'completed.json',{'complete':True,'conditions':m['conditions'],'finals':pairs,'cash_actual_or_storage_allowance_usd':str(actual),'manifest_sha256':sha(HERE/'manifest.json')})
  except BaseException as e:
   save(OUT/'failure.json',{'type':type(e).__name__,'message':str(e),'no_automatic_replay':True});raise
  finally:
   for backend in [base,trained,*evaluation_backends]:backend.close(status='success'if(OUT/'completed.json').exists()else'errored')
if __name__=='__main__':
 {'prepare':prepare,'execute':execute}[sys.argv[1]]()
