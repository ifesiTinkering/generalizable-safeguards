"""Synchronous-policy GRPO; independent sampling/judging is concurrent."""
import hashlib
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from protocol import read, save
from data.data_recipe import next_batch, training_sampling_seed
from data.deterministic_queue import DeterministicQueue, SCHEMA as QUEUE_SCHEMA
from science_stop import patience_status, should_stop, validation_due
from reward_pipeline import score_all, MALFORMED_REWARD, reward_values
from astra_judge import JudgeUnavailable,canonical
from update import build_update
from validation_protocol import observe,update_pins
from validation_metric_adapter import metric_values
from evidence_judge_fallback import ReplaceRow,PolicyRefusal
from student_attempts import sample_three
from data.deterministic_queue import effective_domain
from data.appended_queue import AppendedQueue, SCHEMA as APPEND_SCHEMA

from milestone_pins import MILESTONES, checkpoint_name, pin as pin_milestone

class ValidationIncomplete(RuntimeError):pass

class Engine:
    def __init__(self, config, job, out, backend, prompts, decode, judge, rows, sample_pool, judge_pool):
        self.config,self.job,self.out,self.backend=config,job,Path(out),backend
        self.prompts,self.decode,self.judge,self.rows=prompts,decode,judge,rows
        self.sample_pool,self.judge_pool=sample_pool,judge_pool
        self.path=self.out/'state.json'
        self.state=read(self.path) if self.path.exists() else None
        self.queue = DeterministicQueue(rows['train'], rows_per_domain=config.get('rows_per_domain', 1)) if config.get('row_queue_schema') == QUEUE_SCHEMA else None
        if config.get('row_queue_schema') == APPEND_SCHEMA:
            self.queue = AppendedQueue([r for r in rows['train'] if r.get('supplement_source') != 'policytraj'], [r for r in rows['train'] if r.get('supplement_source') == 'policytraj'])
        if self.queue and self.state and 'queue_cursor' not in self.state:
            raise RuntimeError('Old shuffled cursor cannot be adopted by deterministic queue')
        if self.state and self.state.get('pending'):raise RuntimeError('Uncertain operation; no replay')

    def initialize(self):
        if self.state is not None:raise RuntimeError('Fresh run already initialized')
        self.state={'attempt':0,'updates':0,'pool_pass':0,'next_batch':0,'pending':'initial_checkpoint','history':[],'status':'running'}
        if self.queue: self.state['queue_cursor'] = self.queue.initial_cursor()
        save(self.path,self.state)
        pair=self.backend.checkpoint('initial');self.backend.set_sampling_path(pair['sampling_path'])
        self.state.update(pair,pending=None,pair_registry={pair['checkpoint']:pair});save(self.path,self.state)

    def sample_rows(self, rows, split, attempt, n):
        pending=[];outputs={row['row_id']:[] for row in rows};errors=[]
        for row in rows:
            key=hashlib.sha256(row['row_id'].encode()).hexdigest()
            seed=training_sampling_seed(self.job['seed'],split,row['row_id'],attempt)
            for i in range(n):
                rid=f"{self.config['study_id']}:{self.job['job_id']}:{split}:{attempt}:{key}:{i}"
                f=self.sample_pool.submit(sample_three,self.backend,self.prompts[row['row_id']],seed=(seed+i)%(2**31),max_tokens=self.config['max_tokens'],request_id=rid)
                pending.append((row,i,key,(seed+i)%(2**31),rid,f))
        judged=[]
        for row,i,key,seed,rid,f in pending:
            try:
                seq=f.result()
                if not seq['tokens'] or len(seq['tokens'])!=len(seq['logprobs']) or not all(math.isfinite(v) for v in seq['logprobs']):raise RuntimeError('Sample/logprob mismatch')
                text,clean=self.decode(seq['tokens'])
                # Every arm receives identical semantic validation; only C/D use it for training reward.
                judge=self.judge if split=='validation' or self.job['arm'] in ('C','D') else None
                def grade(row=row,text=text,judge=judge,clean=clean):
                    try:return score_all(row,text,self.config['target'],judge,clean,evidence_requested=True)
                    except JudgeUnavailable as exc:return {'valid':None,'judge_unavailable':True,'reason':str(exc),'rewards':{self.job['arm']:None}}
                future=self.judge_pool.submit(grade)
                raw={'row_id':row['row_id'],'text':text,'clean_termination':clean,'tokens':seq['tokens'],'token_logprobs':seq['logprobs'],'sampling_seed':seed,'request_id':rid,'sample':i,'prompt_sha256':hashlib.sha256(str(self.prompts[row['row_id']]).encode()).hexdigest()}
                save(self.out/'raw_samples'/split/str(attempt)/f'{key}_{i}.json',raw)
                judged.append((row,i,key,seq,raw,future))
            except BaseException as exc:errors.append(exc)
        for row,i,key,seq,raw,f in judged:
            try:
                score=f.result();score.update(raw,reward=score['rewards'][self.job['arm']])
                save(self.out/'raw'/split/str(attempt)/f'{key}_{i}.json',score)
                outputs[row['row_id']].append((seq,score))
            except BaseException as exc:errors.append(exc)
        if errors:raise errors[0]
        return outputs

    def keep(self):
        registry=self.state.get('pair_registry',{})
        ids={self.state['checkpoint']}|{p['checkpoint'] for p in self.state.get('metric_pins',{}).values()}
        if self.state.get('final_pair'):ids.add(self.state['final_pair']['checkpoint'])
        ids.update(p['checkpoint']for p in self.state.get('pending_validation_pairs',{}).values())
        ids.update(p['checkpoint'] for p in self.state.get('dataset_transition_pins', {}).values())
        ids.update(p['checkpoint'] for p in self.state.get('milestone_pins',{}).values())
        missing=ids-set(registry)
        if missing:raise RuntimeError('Missing owned checkpoint pair for durable pin')
        return [registry[k]for k in sorted(ids)]

    def pin_and_prune(self):
        if self.state.get('pending'):raise RuntimeError('Cannot prune with incomplete operation')
        keep=self.keep()
        # The complete metrics and new pins are durable before any remote deletion.
        save(self.out/'retention_pins.json',{'metric_pins':self.state.get('metric_pins',{}),'keep_pairs':keep,'attempt':self.state['attempt'],'all_metric_reports_complete':True})
        final=self.state.get('final_pair',{}).get('checkpoint')
        permanent_ids={p['checkpoint']for p in self.state.get('milestone_pins',{}).values()}|{p['checkpoint']for p in self.state.get('dataset_transition_pins',{}).values()}
        renewable=[p for p in keep if p['checkpoint']!=final and p['checkpoint'] not in permanent_ids]
        if hasattr(self.backend,'renew_archives'):self.backend.renew_archives(renewable)
        else:
            for pair in renewable:self.backend.retain(pair,604800)
        self.backend.prune(keep)

    def evaluate(self, reason='baseline', shared_records=None):
        if shared_records is None and self.state.get('last_validated_checkpoint')==self.state['checkpoint']:
            return read(self.out/f"validation_{self.state['attempt']:06d}"/'records.json')
        self.state['pending']='validation';save(self.path,self.state)
        if shared_records is None:
            outputs=self.sample_rows(self.rows['validation'],'validation',self.state['attempt'],1)
            records=[outputs[r['row_id']][0][1] for r in self.rows['validation']]
        else:
            if self.state['attempt']!=0:raise RuntimeError('Only identical initial SFT validation can be shared')
            if len(shared_records)!=len(self.rows['validation']):raise RuntimeError('Shared population mismatch')
            by_id={r['row_id']:r for r in self.rows['validation']};records=[]
            for r in shared_records:
                if r['row_id'] not in by_id or r['sampling_seed']!=training_sampling_seed(self.job['seed'],'validation',r['row_id'],0) or r['prompt_sha256']!=hashlib.sha256(str(self.prompts[r['row_id']]).encode()).hexdigest():raise RuntimeError('Shared prompt/seed mismatch')
                records.append(dict(r,reward=r['rewards'][self.job['arm']],shared_initial_sft=True))
        dest=self.out/f"validation_{self.state['attempt']:06d}"
        save(dest/'records.json',records)
        packet=metric_values(records,self.rows['validation'])
        save(dest/'all_metrics.json',packet)
        if not packet['complete']:
            self.state['validation_incomplete']=packet['incomplete_reasons'];save(self.path,self.state)
            raise ValidationIncomplete('Validation incomplete; no patience failure, no optimizer advance, no unknown-call replay')
        baseline=self.state['updates']==0 and not self.state['history']
        grid=bool(self.state['updates'] and self.state['updates']%self.config['validation_every_updates']==0 and not any(h['grid']and h['updates']==self.state['updates']for h in self.state['history']))
        self.state['patience_status']=observe(self.state.get('patience_status'),packet['stop_metrics'],complete=True,scheduled=grid,baseline=baseline,delta=self.config['min_delta'],patience=self.config.get('patience'),checkpoint_update=self.state['updates'])
        self.state['pair_registry'][self.state['checkpoint']]={k:self.state[k]for k in('checkpoint','sampling_path')}
        self.state['metric_pins']=update_pins(self.state.get('metric_pins'),self.state['checkpoint'],packet['tracked_metrics'],packet['directions'],complete=True)
        h={'attempt':self.state['attempt'],'updates':self.state['updates'],'grid':grid,'reason':reason,'stop_metrics':packet['stop_metrics'],'populations':packet['populations'],'selection_target':'per_metric_validation_best_no_test_selection','patience_status':self.state['patience_status']}
        h['legacy_partial_composite_diagnostic']=sum(.9*r['legacy_partial']-.5*(r['confidence']-r['legacy_partial'])**2 if r['valid'] else MALFORMED_REWARD for r in records)/len(records)
        h['legacy_exact_composite_diagnostic']=sum(.9*r['exact']-.5*(r['confidence']-r['exact'])**2 if r['valid'] else MALFORMED_REWARD for r in records)/len(records)
        self.state['history'].append(h)
        self.state['last_validated_checkpoint']=self.state['checkpoint']
        self.state['pending']=None;self.state.pop('validation_incomplete',None)
        save(dest/'summary.json',h);save(self.path,self.state)
        self.pin_and_prune()
        return records

    def planned_batch(self):
        if self.queue:
            return self.queue.next_batch(self.state['queue_cursor'])
        return next_batch(self.rows['train'], self.job['seed'], self.state['pool_pass'], self.state['next_batch'])

    def replacement_row(self, cursor, domain):
        if not self.queue: raise RuntimeError('Replacement requires deterministic queue')
        return self.queue.take_domain(cursor, domain)

    def pin_before_supplement(self, row):
        if row.get('supplement_source') != 'policytraj': return
        domain = effective_domain(row)
        pins = self.state.setdefault('dataset_transition_pins', {})
        # First checkpoint is before ANY supplement update, including when a
        # replacement crosses a domain boundary during a prepared batch.
        if 'before_any_policytraj' not in pins:
            pins['before_any_policytraj'] = {k:self.state[k] for k in ('checkpoint','sampling_path')}
            pins['before_any_policytraj'].update(updates=self.state['updates'], attempt=self.state['attempt'], queue_cursor=self.state['queue_cursor'])
        if domain not in pins:
            pins[domain] = {k:self.state[k] for k in ('checkpoint','sampling_path','updates','attempt','queue_cursor')}
            pins[domain]['first_row_id'] = row['row_id']
            save(self.path, self.state)
            save(self.out/'dataset_transition_pins.json', pins)
        for key in ('before_any_policytraj',domain):
            entry=pins[key]
            if not entry.get('permanent_retention_confirmed'):
                rid='permanent-transition:'+hashlib.sha256(entry['checkpoint'].encode()).hexdigest()
                self.backend.retain(entry,None,request_id=rid)
                entry['permanent_retention_confirmed']=True;entry['expires_at']=None
                save(self.path,self.state);save(self.out/'dataset_transition_pins.json',pins)

    def prepare_step(self):
        if self.state.get('pending') or self.state['status']!='running' or should_stop(self.state,self.config):raise RuntimeError('Job not eligible for step')
        batch,cursor=self.planned_batch()
        for row in batch: self.pin_before_supplement(row)
        attempt=self.state['attempt']+1
        self.state['pending']=f'training_attempt_{attempt}';save(self.path,self.state)
        import threading
        output={};selected=[None]*len(batch);replacements=[];replacement_lock=threading.Lock()
        def obtain(index,original):
            nonlocal cursor
            row=original
            for replacement_index in range(self.config.get('max_domain_replacements',140)):
                try:
                    result=self.sample_rows([row],'train',attempt,self.config['samples_per_prompt'])
                    with replacement_lock:output.update(result);selected[index]=row
                    return
                except ReplaceRow as exc:
                    with replacement_lock:
                        replacement,cursor=self.replacement_row(cursor,effective_domain(row))
                        self.pin_before_supplement(replacement)
                        replacements.append({'original_row_id':original['row_id'],'failed_row_id':row['row_id'],'replacement_row_id':replacement['row_id'],'domain':effective_domain(row),'reason':type(exc).__name__+': '+str(exc),'unknown_holds_preserved':True})
                        save(self.out/f'attempt_{attempt:06d}'/'row_replacements.json',{'events':replacements,'proposed_cursor':cursor})
                    row=replacement
            raise RuntimeError('Entire domain replacement pass unavailable; circuit breaker, not scientific convergence')
        with ThreadPoolExecutor(max_workers=len(batch))as row_pool:
            futures=[row_pool.submit(obtain,i,row)for i,row in enumerate(batch)]
            errors=[]
            for future in futures:
                try:future.result()
                except BaseException as exc:errors.append(exc)
            if errors:raise errors[0]
        batch=selected
        groups=[{'prompt_tokens':self.prompts[r['row_id']],'sequences':[s for s,v in output[r['row_id']]],'rewards':[v['reward'] for s,v in output[r['row_id']]]} for r in batch]
        unavailable={r['row_id'] for r in batch if any(v.get('judge_unavailable') for seq,v in output[r['row_id']])}
        for r in batch:
            digest=hashlib.sha256(canonical({k:r[k] for k in ('conversation','policy','policy_framework')}).encode()).hexdigest()
            if hasattr(self.judge,'root') and (self.judge.root/'judge_unavailable_contexts'/digest).exists():unavailable.add(r['row_id'])
        self.prepared_step=(batch,cursor,attempt,output,groups)
        return unavailable

    def commit_step(self,shared_unavailable):
        batch,cursor,attempt,output,groups=self.prepared_step
        masked=[]
        for row,group in zip(batch,groups):
            if row['row_id'] in shared_unavailable:
                masked.append(row['row_id'])
                group['rewards']=[0.]*len(group['sequences']) # zero-gradient placeholder, not a judged reward
        prepared=build_update(groups);prepared['logs']['unavailable_judge_rows']=masked
        dest=self.out/f'attempt_{attempt:06d}' 
        save(dest/'rollouts.json',[v for r in batch for s,v in output[r['row_id']]])
        save(dest/'update_inputs.json',prepared)
        updated=bool(prepared['records'])
        if updated:
            result=self.backend.update(prepared,request_id=f"{self.config['study_id']}:{self.job['job_id']}:update:{attempt}")
            save(dest/'optimizer_metrics.json',result);self.state['updates']+=1
        name=checkpoint_name(self.config,self.job,self.state['updates'],attempt,updated,self.state.get('milestone_pins',{}))
        permanent=updated and self.state['updates'] in MILESTONES and str(self.state['updates']) not in self.state.get('milestone_pins',{})
        pair=self.backend.checkpoint(name,permanent=True) if permanent else self.backend.checkpoint(name)
        self.backend.set_sampling_path(pair['sampling_path'])
        self.state.update(pair,attempt=attempt,pending=None)
        if self.queue:
            self.state['queue_cursor'] = cursor
        else:
            self.state.update(pool_pass=cursor['pool_pass'],next_batch=cursor['next_batch'])
        self.state['pair_registry'][pair['checkpoint']]=pair
        pin_milestone(self.state,self.job,self.config['target'],updated)
        save(self.out/'milestone_pins.json',self.state.get('milestone_pins',{}))
        save(dest/'training_summary.json',{'attempt':attempt,'updates':self.state['updates'],'row_ids':[r['row_id'] for r in batch],'update_logs':prepared['logs']})
        save(self.path,self.state)
        if validation_due(self.state,updated=updated,pass_end=cursor.get('last', False),config=self.config):
            if hasattr(self,'validation_manager'):self.validation_manager.enqueue(self)
            else:raise RuntimeError('v12 requires asynchronous validation manager')
        stop=should_stop(self.state,self.config)
        if stop:self.state['stop_admission_reason']=stop;save(self.path,self.state)
        self.pin_and_prune()
        del self.prepared_step
        return self.state

    def step(self):
        return self.commit_step(self.prepare_step())

    def finish(self, reason):
        if hasattr(self,'validation_manager'):
            if self.validation_manager.has_pending(self):raise RuntimeError('Drain admitted validation before final retention')
        else:raise RuntimeError('v12 requires asynchronous validation manager')
        if self.state.get('pending'):raise RuntimeError('Cannot finish with incomplete validation/operation')
        self.state['pending']='final_checkpoint';save(self.path,self.state)
        pair=self.backend.checkpoint('final',final=True)
        self.state['pair_registry'][pair['checkpoint']]=pair
        self.state.update(pair,final_pair=pair,status='finalizing',stop_reason=reason,pending=None)
        save(self.path,self.state)
        self.pin_and_prune()
        self.state['status']='complete';save(self.path,self.state)
