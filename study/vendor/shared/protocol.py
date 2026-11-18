"""Evidence-run authorization, immutable source binding and atomic state writes."""
import hashlib
import json
import os
import uuid
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def require_authorized(*args):
    from supplement_protocol import require_authorized as check
    return check()
