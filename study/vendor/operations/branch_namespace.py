"""Every new branch operation gets a fresh logical ID, including checkpoint saves."""
import hashlib
from pathlib import Path
EXPECTED='12cec41ec422de8c76309e14e2a156891fb1e5aff100ad1463145c1328d36c4b'
def install(module):
 if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Unreviewed backend')
 cls=module.LiveBackend
 if getattr(cls,'_claim2_branch_namespace',False):raise ValueError('Branch namespace already installed')
 original=cls._execute
 def execute(self,request_id,*args,**kwargs):
  namespace=self.config.get('study_id')
  if not isinstance(namespace,str)or not namespace.startswith('evidence:'):raise ValueError('Explicit recovery branch namespace required')
  return original(self,namespace+':branch:'+request_id,*args,**kwargs)
 cls._execute=execute;cls._claim2_branch_namespace=True
