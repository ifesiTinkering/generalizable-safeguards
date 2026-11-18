"""Once-only cross-entropy transport, inherited budget/journal/no-retry rules."""
import hashlib
import json
from live_backend import LiveBackend, disable_retries
from cost_plan import training_upper_bound


class SFTBackend(LiveBackend):
    def supervised_update(self, records, request_id):
        if not records: raise ValueError('Empty SFT batch')
        datums=[]
        for r in records:
            n=len(r['input_tokens'])
            if n<1 or len(r['target_tokens'])!=n or len(r['weights'])!=n:
                raise ValueError('SFT alignment mismatch')
            if not any(r['weights']) or any(w not in (0.,1.) for w in r['weights']):
                raise ValueError('SFT requires nonempty binary loss mask')
            datums.append(self.sdk.Datum(model_input=self.sdk.ModelInput.from_ints(r['input_tokens']),
                loss_fn_inputs={k:self.sdk.types.TensorData(data=r[k],dtype=d,shape=[n])
                                for k,d in [('target_tokens','int64'),('weights','float32')]}))
        upper=training_upper_bound([len(r['input_tokens'])+1 for r in records])
        spec={'kind':'update','objective':'format_sft','loss_fn':'cross_entropy','reduction':'weighted_sum',
              'training_tokens_upper_bound':sum(len(r['input_tokens'])+1 for r in records),
              'records_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
              'learning_rate':self.config['learning_rate'],'adam':self.config['adam']}
        def invoke():
            disable_retries(self.client)
            backward=self.client.forward_backward(datums,loss_fn='cross_entropy')
            step=self.client.optim_step(self.sdk.AdamParams(learning_rate=self.config['learning_rate'],**self.config['adam']))
            b,s=backward.result(),step.result()
            return {'backward_metrics':getattr(b,'metrics',{}),'optimizer_metrics':getattr(s,'metrics',{}),
                    'loss_fn_outputs':getattr(b,'loss_fn_outputs',[]),'loss_reduction':'weighted_sum',
                    'charged_full_sequence_upper_bound':str(upper)}
        return self._execute(request_id,spec,upper,invoke,allow_cached=False)
