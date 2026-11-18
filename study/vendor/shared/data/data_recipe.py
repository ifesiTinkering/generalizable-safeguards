"""Isolated Claim 2 data-only copy of Claim 3's frozen recipe. No model calls.

Rows and batches.py are byte-for-byte copies. load_rows/next_batch preserve the
Claim 3 behavior with BASE redirected here. The seed namespace is deliberately
unchanged to match the exact Claim 3 sampling seeds, not merely their recipe.
Original scripts in original_reference are provenance only; do not execute them.
"""
import hashlib,json,importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
def load_rows(stage):
 import pyarrow.parquet as pq
 if stage not in ('train','validation','test_id','test_ood','sft'):raise ValueError('Unknown data role')
 manifest=json.loads((BASE/'data_manifest.json').read_text());m=manifest['files'][stage]
 p=BASE/m['path']
 if hashlib.sha256(p.read_bytes()).hexdigest()!=m['sha256']:raise RuntimeError('Frozen data changed')
 rows=pq.read_table(p).to_pylist()
 if len(rows)!=manifest['row_totals'][stage] or len(rows)!=len({r['row_id'] for r in rows}):raise RuntimeError('Invalid frozen rows')
 return rows
def next_batch(rows,seed,pass_index,batch_index):
 if pass_index<0 or batch_index<0:raise ValueError('Negative resume cursor')
 spec=importlib.util.spec_from_file_location('claim2_safe_batches',BASE/'batches.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 batches=list(mod.batches(rows,seed+pass_index))
 if batch_index>len(batches):raise ValueError('Resume cursor beyond pass')
 if batch_index==len(batches):
  pass_index+=1;batch_index=0;batches=list(mod.batches(rows,seed+pass_index))
 return batches[batch_index],{'pool_pass':pass_index,'next_batch':batch_index+1,'completed_passes':pass_index+int(batch_index+1==len(batches)),'last':batch_index+1==len(batches)}
def sample_seed(root,experiment,split,row_id,attempt=None):
 payload=[root,experiment,split,row_id,attempt if split=='grpo' else None]
 return int(hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:8],16)%(2**31)
def training_sampling_seed(root,split,row_id,attempt):
 return sample_seed(root,'claim3_safepyramid_large_v2','grpo' if split=='train' else split,row_id,attempt if split=='train' else 0)
def final_sampling_seed(split,row_id):
 if split not in ('test_id','test_ood'):raise ValueError('Final split required')
 return sample_seed(101,'claim3_safepyramid_large_v2',split,row_id)
