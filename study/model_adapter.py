"""Native OFF framing with no prompt truncation or completion repair."""
import threading

def prepare(cfg,rows):
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers
    from objective_prompts import messages
    from tinker_cookbook.renderers import get_text_content
    name='qwen3_5_disable_thinking' if cfg['model_family']=='qwen' else 'gpt_oss_no_sysprompt'
    renderer=renderers.get_renderer(name,AutoTokenizer.from_pretrained(cfg['tokenizer_path'],local_files_only=True))
    lock=threading.Lock()
    def prompt(row):
        msg=messages(row,cfg['target'],cfg['arm'])
        if cfg['model_family']=='gpt':
            from gpt_modes import prompt
            return prompt(renderer,msg,'OFF')
        return renderer.build_generation_prompt(msg,prefill='<answer>\n').to_ints()
    prompts={r['row_id']:prompt(r)for group in rows.values()for r in group}
    if any(len(p)+cfg['max_tokens']>cfg['context_limit']for p in prompts.values()):
        raise ValueError('Context overflow; source rows are never truncated')
    def decode(tokens):
        with lock:
            if cfg['model_family']=='gpt':
                from gpt_modes import decode_text
                result=decode_text(renderer.tokenizer.decode(tokens,skip_special_tokens=False),'OFF')
                return result['text'],result['clean']
            message,end=renderer.parse_response(tokens)
            return '<answer>\n'+get_text_content(message),end.is_clean
    return renderer,prompts,decode
