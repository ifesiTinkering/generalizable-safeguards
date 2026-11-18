"""Append-only PolicyTraj queue; migrate each run's own original cursor."""
import copy,hashlib,json
from data.deterministic_queue import DeterministicQueue, DOMAINS, effective_domain
SCHEMA='claim2-safepyramid-policytraj-append-v1'
class AppendedQueue:
 def __init__(self,original,supplement):
  self.original=DeterministicQueue(original,1); self.supplement={d:[]for d in DOMAINS}
  seen={r['row_id']for r in original}
  for r in supplement:
   if r['row_id']in seen:raise ValueError('Duplicate source row')
   seen.add(r['row_id']);self.supplement[effective_domain(r)].append(r)
  if any(not v for v in self.supplement.values()):raise ValueError('Empty supplement domain')
  self.queue_sha256=hashlib.sha256(json.dumps({'original':self.original.queue_sha256,'supplement':self.supplement},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 def migrate(self,cursor):
  # Validate against old schema/hash without consuming the result.
  self.original.take_domain(cursor,DOMAINS[0])
  offsets=dict(cursor['offsets'])
  # Finish the CURRENT original pass, preserving the exact next original row.
  ends={d:((n//len(self.original.rows[d])+1)*len(self.original.rows[d]) if n%len(self.original.rows[d]) else n) for d,n in offsets.items()}
  ends={d:(n if n else len(self.original.rows[d]))for d,n in ends.items()}
  return dict(schema=SCHEMA,queue_sha256=self.queue_sha256,original_offsets=offsets,original_end_offsets=ends,supplement_offsets={d:0 for d in DOMAINS},append_phase_offsets={d:0 for d in DOMAINS},migration_source_cursor=copy.deepcopy(cursor))
 def take_domain(self,cursor,domain):
  if cursor.get('schema')!=SCHEMA or cursor.get('queue_sha256')!=self.queue_sha256:raise ValueError('Wrong appended queue cursor')
  c=copy.deepcopy(cursor)
  if c['original_offsets'][domain]<c['original_end_offsets'][domain]:
   row=self.original.rows[domain][c['original_offsets'][domain]%len(self.original.rows[domain])];c['original_offsets'][domain]+=1
  else:
   # First finish supplement, then repeat complete original+supplement cycles.
   added=self.supplement[domain];original=self.original.rows[domain]
   pos=c['append_phase_offsets'][domain] % (len(added)+len(original))
   if pos<len(added):
    row=added[pos];c['supplement_offsets'][domain]+=1
   else:
    row=original[pos-len(added)]
   c['append_phase_offsets'][domain]+=1
  return row,c
 def next_batch(self,cursor):
  rows=[]
  for domain in DOMAINS:
   row,cursor=self.take_domain(cursor,domain);rows.append(row)
  return rows,cursor
 def is_supplement(self,row):return row.get('supplement_source')=='policytraj'
