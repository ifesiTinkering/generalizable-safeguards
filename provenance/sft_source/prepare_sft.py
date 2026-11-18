"""Offline GPT tokenization of the preserved Qwen SFT curriculum; no API calls."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    'initial': ('claim2_format22_pilot_20261112', 'rows32.json'),
    'extension': ('claim2_sft_extension55_20261112', 'new23_rows.json'),
    'format': ('claim2_format_only55_20261112', 'training_rows.json'),
}

def main(stage):
    folder, filename = STAGES[stage]
    source = ROOT / 'sft_source_snapshot' / folder
    sys.path[:0] = [str(source), str(ROOT / 'source_snapshot')]
    from target_builder_gpt import build_record
    from objective_prompts import messages
    from reward_pipeline import parse
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'tokenizer', local_files_only=True)
    renderer = renderers.get_renderer('gpt_oss_no_sysprompt', tokenizer)
    rows = json.loads((source / filename).read_text())
    datums = {}
    for row in rows:
        datum = build_record(renderer, messages(row, 'iou', 'D'), row)
        parsed = parse(datum['audit']['target'], [str(r['rule_id']) for r in row['all_policy_rules']])
        assert set(parsed['violated_rule_ids']) == set(row['ground_truth_rule_ids'])
        assert len(datum['input_tokens']) == len(datum['target_tokens']) == len(datum['weights'])
        datums[row['row_id']] = datum
    out = ROOT / 'prepared_sft'
    out.mkdir(exist_ok=True)
    serialized = json.dumps(datums, ensure_ascii=False).encode()
    (out / (stage + '_datums.json')).write_bytes(serialized)
    receipt = {'stage': stage, 'rows': len(rows), 'provider_calls': 0,
               'sha256': hashlib.sha256(serialized).hexdigest(),
               'max_sequence_tokens': max(len(d['input_tokens']) + 1 for d in datums.values()),
               'target_mask_policy': 'Preserved stage-specific character spans; GPT tokenizer offsets and return token',
               'passed': True}
    (out / (stage + '_receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=STAGES)
    main(parser.parse_args().stage)
