"""23newrows,18fixed updates continuing the30positive optimizer; versioned maskfix."""
import sys,os,json,time,hashlib,threading,random,fcntl
from pathlib import Path
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
PILOT=HERE.parent/'claim2_format22_pilot_20261112'
CODE=HERE.parent/'runtime_v9_tinker_template'
sys.path[1:1]=[str(PILOT),str(CODE)]
from protocol import read,save,sha
from target_builder import build_record
from objective_prompts import messages
from cost_plan import training_upper_bound,sampling_upper_bound
from data.data_recipe import final_sampling_seed
from thinking_adapter import renderer,decode_with_audit
from reward_pipeline import parse,score_all
from heldout_metric_adapter import metric_values
ROOT=Path('/workspace/safepyramid-runs/claim2_evidence_ablations_20261111/partial')
PRIOR=ROOT/'pilot_sft_learningcurve_20261112'
OUT=ROOT/'pilot_sft_extension55_20261112'

def paused():
 for pid in(973394,973395):
  if '\nState:\tT'not in Path(f'/proc/{pid}/status').read_text():raise RuntimeError('Old48mustremainpaused')
def batches(rows):
 result=[]
 for seed in(0,1,2):
  rr=sorted(rows,key=lambda x:x['row_id']);random.Random(seed).shuffle(rr);result.extend([rr[i:i+4]for i in range(0,len(rr),4)])
 return result

def lineage():
 m=read(PILOT/'manifest.json')
 for p,h in m['source_hashes'].items():
  if sha(p)!=h:raise RuntimeError('Originalpilotsourcechanged')
 state=read(PRIOR/'state.json')
 if state['updates']!=27 or state['pending']is not None:raise RuntimeError('Prioroptimizer not durablyfinished27')
 js=[read(p)for p in(PRIOR/'continuous_training/transport_journal').glob('*.json')]
 if sum(j.get('spec',{}).get('kind')=='update'and j['status']=='complete'for j in js)!=27:raise RuntimeError('Need27completeoptimizerjournals')
 pair=read(PRIOR/'format32/final_checkpoint.json')
 if not any(j.get('spec',{}).get('kind')=='save_state'and j['status']=='complete'and j.get('result',{}).get('path')==pair['checkpoint']for j in js):raise RuntimeError('Parentcheckpoint provenance mismatch')
 return pair

def prepare():
 parent=lineage();rows=read(HERE/'new23_rows.json');prior=read(PILOT/'rows32.json');allrows=read(HERE/'cumulative55_rows.json')
 assert len(rows)==23 and sum(bool(r['ground_truth_rule_ids'])for r in rows)==20
 assert len(allrows)==55 and sum(bool(r['ground_truth_rule_ids'])for r in allrows)==50
 assert not({r['row_id']for r in rows}&{r['row_id']for r in prior})
 assert {r['row_id']for r in allrows}=={r['row_id']for r in rows+prior}
 r=renderer(False);datums={x['row_id']:build_record(r,messages(x,'partial','A'),x)for x in rows}
 for row in rows:
  parsed=parse(datums[row['row_id']]['audit']['target'],[str(x['rule_id'])for x in row['all_policy_rules']]);assert set(parsed['violated_rule_ids'])==set(row['ground_truth_rule_ids'])
 heldout=read(PILOT/'validation_rows.json');assert not({x['conversation_sha256']for x in rows}&{x['conversation_sha256']for x in heldout})
 prompts=read(PILOT/'validation_prompts.json');trainbound=training_upper_bound([len(datums[x['row_id']]['input_tokens'])+1 for b in batches(rows)for x in b]);evalbound=sum((sampling_upper_bound(len(p),10000 if mode=='ON'else 23000)for mode,ps in prompts.items()for p in ps.values()),Decimal(0));bound=trainbound+evalbound+Decimal('.2');assert bound<20
 save(HERE/'prepared_datums.json',datums)
 sources={str(p):sha(p)for p in HERE.glob('*.py')}
 for name in('new23_rows.json','cumulative55_rows.json','selection_manifest.json','prepared_datums.json'):sources[str(HERE/name)]=sha(HERE/name)
 m={'source_hashes':sources,'parent_manifest_sha256':sha(PILOT/'manifest.json'),'parent':parent,'parent_updates':27,'restore_optimizer':True,'new_updates':18,'total_updates':45,'new_rows':23,'new_positive':20,'new_negative':3,'cumulative_positive':50,'cumulative_negative':5,'epochs':3,'batch_sizes':[len(b)for b in batches(rows)],'new_exposures':69,'learning_rate':1e-5,'adam':{'beta1':.9,'beta2':.95,'eps':1e-8},'train_upper_usd':str(trainbound),'eval_upper_usd':str(evalbound),'storage_allowance_usd':'.2','full_upper_usd':str(bound),'format_change':'Answer IDs and copied evidence ruleID attributes plus allowedspeaker enum assistant supervised. TurnID1/placeholderquotes/justification/review/q remainmasked. Syntheticmetadata not asserted asgroundedevidence.','comparison_scope':'More rows AND explicit maskfix, not isolated datasizeeffect.','goal':'Measureformatreliability; no guarantee95percent and no forcedF1ceiling','evaluation':'Same488OFF23000/ON10000TOTAL,sameprompts/seeds. MissingsemanticN/A,neverretryblockedjudge.'}
 save(HERE/'manifest.json',m);print(json.dumps(m|{'source_hashes':None},indent=2))

def execute():
 paused();m=read(HERE/'manifest.json');a=read(HERE/'authorization.json')
 if a.get('approved')is not True or a.get('manifest_sha256')!=sha(HERE/'manifest.json'):raise RuntimeError('Independentauditrequired')
 if sha(PILOT/'manifest.json')!=m['parent_manifest_sha256']or lineage()!=m['parent']:raise RuntimeError('Parentlineagechanged')
 for p,h in m['source_hashes'].items():
  if sha(p)!=h:raise RuntimeError('Sourcechanged')
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
  ledger=Ledger(ROOT/'budget.sqlite',cap_usd=1000);judge=AstraJudge(ROOT)
  with judge.ledger.tx()as db:
   cap=db.execute('select max_calls from grant_meta').fetchone()[0];used=db.execute("select count(*)from grant_calls where status!='cancelled'").fetchone()[0]
  if used+3904>cap:raise RuntimeError('Grantcapacity incl otherlivepilotoutputs unavailable')
  meter=CohortMeter(ledger,'evidence:sftcurve:extension55',Decimal(m['full_upper_usd']),OUT/'cohort.json')
  meter.reserve('evidence:sftcurve:extension55:storage',Decimal('.2'),{'kind':'modeled_storage_allowance'});meter.dispatch('evidence:sftcurve:extension55:storage');meter.settle('evidence:sftcurve:extension55:storage',Decimal('.2'))
  b=SFTBackend.__new__(SFTBackend);b.config={'model':'Qwen/Qwen3.5-9B','initial_checkpoint':m['parent']['checkpoint'],'learning_rate':1e-5,'adam':m['adam'],'storage_reserve_confirmed':True,'storage_max_pairs_per_job':1};b.outdir=OUT;b.journal=OUT/'transport_journal';b.journal.mkdir();b.meter=meter;b.sdk=tinker;b.retry_class=RetryConfig;b.service=tinker.ServiceClient();disable_retries(b.service);b.client=None;b.sampler=None;b.sampling_path=None
  try:
   def restore():
    b.client=b.service.create_training_client_from_state_with_optimizer(m['parent']['checkpoint']);disable_retries(b.client);return {'checkpoint':m['parent']['checkpoint'],'restore_optimizer':True}
   b._execute('evidence:sftcurve:extension55:restore',{'kind':'restore_optimizer','checkpoint':m['parent']['checkpoint']},Decimal(0),restore,allow_cached=False)
   ds=read(HERE/'prepared_datums.json')
   for index,batch in enumerate(batches(read(HERE/'new23_rows.json')),1):
    paused();save(OUT/'state.json',{'new_updates':index-1,'total_updates':27+index-1,'pending':index});result=b.supervised_update([ds[x['row_id']]for x in batch],'evidence:sftcurve:extension55:update:'+str(index));save(OUT/'updates'/f'{index:03d}.json',result);save(OUT/'state.json',{'new_updates':index,'total_updates':27+index,'pending':None});print(json.dumps({'stage':'extension_update','new_update':index,'total_update':27+index}),flush=True)
   pair=b.checkpoint('format55-milestone');b.retain(pair,604800);save(OUT/'final_checkpoint.json',pair)
   rows=read(PILOT/'validation_rows.json');prompts=read(PILOT/'validation_prompts.json');decode_lock=threading.Lock()
   for mode in('OFF','ON'):
    r=renderer(mode=='ON');b.config['stop_sequences']=r.get_stop_sequences();limit=10000 if mode=='ON'else 23000;dest=OUT/'evaluations'/mode;dest.mkdir(parents=True,exist_ok=True)
    def one(row):
     seed=final_sampling_seed(row['split'],row['row_id']);seq=b.sample(prompts[mode][row['row_id']],seed,limit,'evidence:sftcurve:extension55:'+mode+':'+row['row_id'])
     with decode_lock:decoded=decode_with_audit(r,seq['tokens'],thinking=mode=='ON')
     raw={'row_id':row['row_id'],'split':row['split'],'mode':mode,'text':decoded['text'],'clean_termination':decoded['clean'],'sampling_seed':seed,'tokens':seq['tokens'],'token_logprobs':seq['logprobs'],'decoder_audit':decoded,'prompt_sha256':hashlib.sha256(json.dumps(prompts[mode][row['row_id']]).encode()).hexdigest(),'sampling_config':{'thinking':mode=='ON','temperature':1.0,'max_tokens':limit}}
     key=hashlib.sha256(row['row_id'].encode()).hexdigest();save(dest/'raw'/(key+'.json'),raw)
     try:scores=score_all(row,raw['text'],'partial',judge,raw['clean_termination'])
     except JudgeUnavailable as exc:scores=dict(score_all(row,raw['text'],'partial',None,raw['clean_termination']),judge_unavailable=True,reason=str(exc),correctness_derived_offline=True)
     record=raw|scores;save(dest/'scored'/(key+'.json'),record);return record
    with ThreadPoolExecutor(max_workers=32)as pool:records=list(pool.map(one,rows))
    save(dest/'records.json',records);packet=metric_values(records,rows);save(dest/'metrics.json',packet);receipt={'sampling_complete':True,'correctness_complete':True,'semantic_complete':packet['complete'],'n':488,'records_sha256':sha(dest/'records.json'),'semantic_missing_rows':[x['row_id']for x in records if x.get('judge_unavailable')]};save(dest/'sampling_complete.json',receipt)
    if packet['complete']:save(dest/'complete.json',dict(receipt,complete=True))
   actual=meter.close();save(OUT/'completed.json',{'sampling_complete':True,'new_updates':18,'total_updates':45,'final':pair,'cash_actual_or_allowance_usd':str(actual),'manifest_sha256':sha(HERE/'manifest.json')})
  except BaseException as exc:save(OUT/'failure.json',{'type':type(exc).__name__,'message':str(exc),'no_retry':True});raise
  finally:b.close(status='success'if(OUT/'completed.json').exists()else'errored')
if __name__=='__main__':{'prepare':prepare,'execute':execute}[sys.argv[1]]()
