"""One model per interpreter; explicit module precedence, no SDK import."""
import os
import sys
from pathlib import Path

_family=None

def install(family):
    global _family
    if family not in ('qwen','gpt'): raise ValueError('Unknown model family')
    if _family is not None and _family!=family: raise RuntimeError('Use separate processes for different model families')
    root=Path(__file__).resolve().parent
    if _family is None:
        sys.path[:0]=[str(root/'portable'),str(root/'vendor'/family),str(root/'vendor/operations'),str(root/'vendor/shared')]
        os.environ['POLICY_STUDY_MODEL_FAMILY']=family
        _family=family
    return root
