"""Shared domain queues: evenly interleave negatives without oversampling.

Preserve frozen-file relative order within each domain/class. Every row occurs
exactly once per pass. All seeds/arms/targets use this same hashed queue; only
completion sampling uses the model seed. Persist offsets before recovery.
"""
import hashlib
import json
DOMAINS = ('privacy', 'fraud', 'defamation')
SCHEMA = 'claim2-level-rotating-domain-queue-v3'
LEVELS = ('L0', 'L1', 'L2')

def effective_domain(row):
    # Negative conversations have no taxonomy domain; target_domain identifies
    # the policy being evaluated. Never mutate the frozen source metadata.
    if not row.get('ground_truth_rule_ids'):
        domain = row.get('target_domain') or row.get('domain')
    else:
        domain = row.get('domain')
    if domain not in DOMAINS:
        raise ValueError('Missing or unknown training policy domain')
    return domain

class DeterministicQueue:
    def __init__(self, rows, rows_per_domain=1):
        if type(rows_per_domain) is not int or rows_per_domain < 1:
            raise ValueError('Positive rows_per_domain required')
        self.rows_per_domain = rows_per_domain
        domains = {d: [] for d in DOMAINS}
        seen = set()
        for row in rows:
            domain, rid = effective_domain(row), row['row_id']
            if domain not in domains or rid in seen:
                raise ValueError('Unexpected domain or duplicate row ID')
            if not isinstance(row.get('ground_truth_rule_ids'), list):
                raise ValueError('Explicit gold rule list required to classify negatives')
            if row.get('native_level') not in LEVELS:
                raise ValueError('Explicit L0/L1/L2 label required')
            seen.add(rid); domains[domain].append(row)
        if any(len(rs) < rows_per_domain for rs in domains.values()):
            raise ValueError('Insufficient unique rows in a domain')
        self.rows = {}; self.level_fallbacks = []
        for domain_index, (domain, items) in enumerate(domains.items()):
            buckets = {(negative, level): [] for negative in (False, True) for level in LEVELS}
            for row in items:
                buckets[(not bool(row['ground_truth_rule_ids']), row['native_level'])].append(row)
            total_negative = sum(not bool(r['ground_truth_rule_ids']) for r in items)
            used_negative = 0; ordered = []
            for position in range(len(items)):
                negative = (position + 1) * total_negative // len(items) > used_negative
                desired_index = (position + domain_index) % 3
                desired = LEVELS[desired_index]
                available = [LEVELS[(desired_index + offset) % 3] for offset in range(3)
                             if buckets[(negative, LEVELS[(desired_index + offset) % 3])]]
                if not available: raise RuntimeError('Class allocation exhausted unexpectedly')
                selected = available[0]
                row = buckets[(negative, selected)].pop(0)
                ordered.append(row); used_negative += int(negative)
                if selected != desired:
                    self.level_fallbacks.append(dict(domain=domain, position=position,
                        requested_level=desired, actual_level=selected, row_id=row['row_id'],
                        reason='requested class/level exhausted; fixed cyclic fallback'))
            self.rows[domain] = ordered
        roster = [(d, [(r['row_id'], r['native_level'], bool(r['ground_truth_rule_ids'])) for r in self.rows[d]]) for d in DOMAINS]
        self.queue_sha256 = hashlib.sha256(json.dumps(roster, separators=(',', ':')).encode()).hexdigest()

    def initial_cursor(self):
        return dict(schema=SCHEMA, queue_sha256=self.queue_sha256, offsets={d: 0 for d in DOMAINS})

    def take_domain(self, cursor, domain):
        if cursor.get('schema') != SCHEMA or cursor.get('queue_sha256') != self.queue_sha256:
            raise ValueError('Incompatible queue cursor or row order')
        offsets = dict(cursor.get('offsets', {}))
        if set(offsets) != set(DOMAINS) or any(type(n) is not int or n < 0 for n in offsets.values()):
            raise ValueError('Invalid domain cursor')
        if domain not in DOMAINS: raise ValueError('Unexpected domain')
        row = self.rows[domain][offsets[domain] % len(self.rows[domain])]
        offsets[domain] += 1
        return row, dict(cursor, offsets=offsets)

    def next_batch(self, cursor):
        batch = []
        for _ in range(self.rows_per_domain):
            for domain in DOMAINS:
                row, cursor = self.take_domain(cursor, domain); batch.append(row)
        return batch, cursor
