"""Once-only Astra requests, durable results and separate grant accounting."""
import fcntl
import hashlib
import json
import os
import uuid
from decimal import Decimal
from pathlib import Path
import httpx
from astra_contract import INSTRUCTIONS, SCHEMA
from evidence_budget import GrantLedger

class JudgeUnavailable(RuntimeError):
    """Unavailable judge result; never treated as evidence of answer quality."""
    pass

def canonical(x): return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
def save(path, value):
    p = Path(path); temp = p.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(canonical(value)); os.replace(temp, p)

class AstraJudge:
    def __init__(self, root, key=None, post=None, max_calls=46440):
        self.root = Path(root); self.cache = self.root/'astra_cache'; self.cache.mkdir(parents=True, exist_ok=True)
        self.ledger = GrantLedger(self.root/'grant_usage.sqlite',max_calls=max_calls)
        self.key = key or os.environ.get('OPENAI_API_KEY')
        self.post = post or httpx.post
        if not self.key: raise RuntimeError('Judge credential unavailable')

    def __call__(self, context, p):
        context_hash=hashlib.sha256(canonical(context).encode()).hexdigest()
        blocked=self.root/'judge_unavailable_contexts'/context_hash
        if blocked.exists():raise JudgeUnavailable('Context previously rejected by provider; no new request')
        candidate = {k: p[k] for k in ('is_violation', 'violated_rule_ids', 'rule_assessments')}
        body = {'model': 'gpt-6-astra', 'store': False, 'instructions': INSTRUCTIONS,
                'input': canonical({**context, 'candidate': candidate}), 'max_output_tokens': 8192,
                'text': {'format': {'type':'json_schema', 'name':'per_rule_support', 'strict':True, 'schema':SCHEMA}}}
        digest = hashlib.sha256(canonical(body).encode()).hexdigest()
        folder = self.cache/digest; folder.mkdir(exist_ok=True)
        with (folder/'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            raw = folder/'response.json'
            if raw.exists():
                response = json.loads(raw.read_text())
            else:
                if not self.ledger.reserve(digest, {'request_sha256':digest, 'model':'gpt-6-astra', 'max_output_tokens':8192}):
                    raise JudgeUnavailable('Prior judge outcome unknown; reservation retained, never replayed')
                save(folder/'request.json', body)
                if not self.ledger.dispatch(digest): raise RuntimeError('Judge dispatch already used')
                try:
                    response_obj = self.post('https://api.openai.com/v1/responses', headers={'Authorization':'Bearer '+self.key}, json=body, timeout=180, trust_env=False)
                except httpx.TransportError as exc:
                    save(folder/'unavailable.json', {'kind':'unknown_transport_outcome','error_type':type(exc).__name__,'reservation_retained':True,'never_replay':True})
                    raise JudgeUnavailable('Judge transport outcome unknown; reservation retained, never replayed') from exc
                try:response = response_obj.json()
                except ValueError as exc:
                    save(folder/'unavailable.json', {'kind':'non_json_response','status':response_obj.status_code,'reservation_retained':True,'never_replay':True})
                    raise JudgeUnavailable('Non-JSON judge response; reservation retained, never replayed') from exc
                save(raw, response)
                save(folder/'http_status.json', {'status':response_obj.status_code})
            error=response.get('error') or {}
            if error.get('code') in ('cyber_policy','content_policy_violation','policy_violation','safety_violation'):
                blocked.parent.mkdir(exist_ok=True)
                save(blocked,{'context_sha256':context_hash,'request_sha256':digest,'error_code':error['code'],'not_retried':True})
                raise JudgeUnavailable('Provider policy rejection; no reward assigned')
            usage = response.get('usage')
            if not usage: raise JudgeUnavailable('Judge usage unknown; retain grant reservation and do not replay')
            # Conservative uncached retail equivalent; grant cash expense is zero.
            economic = (Decimal(usage['input_tokens'])*10+Decimal(usage['output_tokens'])*50)/1000000
            self.ledger.settle_usage(digest, usage, economic)
            if response.get('status') != 'completed': raise JudgeUnavailable('Judge incomplete/refused; not negative evidence; no retry')
            texts = [part['text'] for item in response.get('output',[]) if item.get('type')=='message' for part in item.get('content',[]) if part.get('type')=='output_text']
            if len(texts)!=1: raise JudgeUnavailable('Missing judge structured response; no retry')
            try:value = json.loads(texts[0])
            except ValueError as exc:raise JudgeUnavailable('Malformed judge JSON; no retry') from exc
            if not isinstance(value,dict):raise JudgeUnavailable('Judge response is not an object; no retry')
            values = value.get('rules')
            if not isinstance(values,list) or len(values)!=len(candidate['violated_rule_ids']): raise JudgeUnavailable('Judge response omitted or duplicated requested rules; no reward assigned')
            if any(not isinstance(v,dict) for v in values):raise JudgeUnavailable('Malformed rule verdict; no retry')
            if {v.get('rule_id') for v in values} != set(candidate['violated_rule_ids']): raise JudgeUnavailable('Judge response IDs differ; no reward assigned')
            for v in values:
                if type(v.get('claim_supported')) is not bool or type(v.get('evidence_supported')) is not bool or not isinstance(v.get('reason'),str): raise JudgeUnavailable('Judge response fields invalid; no reward assigned')
                if v['evidence_supported'] and not v['claim_supported']: raise JudgeUnavailable('Judge verdict inconsistent; no reward assigned')
            return {'rules':values, 'request_sha256':digest, 'model':response.get('model'), 'usage':usage}
