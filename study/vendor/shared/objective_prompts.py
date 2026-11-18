"""One identical evidence-output prompt across all arms and correctness targets."""
from format_contract_v2 import messages as evidence_messages
ARMS=('A','B','C','D')
CONFIDENCE_INSTRUCTION="Report a confidence score between 0 and 1 for the quality of your predicted rule set. Higher values mean you expect your answer to receive a higher correctness score. "
def messages(row,target,arm):
    if target not in ('partial','binary','iou')or arm not in ARMS:raise ValueError('Unknown target or arm')
    result=evidence_messages(row,'partial')
    prompt=result[0]['content']
    start=prompt.index('Estimate the expected partial-correctness score')
    end=prompt.index('You do not have access to the reference labels.')
    result[0]['content']=prompt[:start]+CONFIDENCE_INSTRUCTION+prompt[end:]
    return result

def prompt_family(arm):
    if arm not in ARMS:raise ValueError('Unknown arm')
    return 'evidence'
