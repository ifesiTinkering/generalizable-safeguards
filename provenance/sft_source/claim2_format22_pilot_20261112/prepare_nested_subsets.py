import hashlib,json,collections
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
old=json.loads((ROOT/'rows.json').read_text());source=Path('training/safepyramid_experiments/selection_v2/splits/defamation_to_content_moderation/sft/rows.parquet');rows=pq.read_table(source).to_pylist()
key=lambda r:hashlib.sha256(('claim2-format22-v1:'+r['row_id']).encode()).hexdigest()
neg=[r for r in old if not r['is_violation']];pos20=[r for r in old if r['is_violation']];assert len(neg)==2 and len(pos20)==20
selected={}
for count,levels in [(10,[4,3,3]),(20,[7,7,6]),(30,[10,10,10])]:
 pos=[]
 for level,n in zip(['L0','L1','L2'],levels):
  pool=pos20 if count==10 else [r for r in rows if r['is_violation']]
  pos.extend(sorted([r for r in pool if r['native_level']==level],key=key)[:n])
 chosen=sorted(pos+neg,key=lambda r:r['row_id']);selected[count]={r['row_id'] for r in pos}
 if count==20:assert {r['row_id'] for r in chosen}=={r['row_id'] for r in old}
 dest=ROOT/f'subset_positive{count}';dest.mkdir(exist_ok=True)
 raw=json.dumps(chosen,indent=2,ensure_ascii=False,default=str)+'\n';(dest/'rows.json').write_text(raw)
 manifest={'condition':f'positive{count}','positive_count':count,'negative_count':2,'total_rows':count+2,'row_ids':[r['row_id'] for r in chosen],'row_sha256':hashlib.sha256(raw.encode()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selection':'Nested stable-hash strata; originalselected20and2negative unchanged; no modeloutcomes','initialization':'Independent fresh rawbase for EACH condition; not sequential continuation','epochs':3,'learning_rate':.00001,'batch_size':4,'updates':3*((count+2+3)//4),'row_exposures':3*(count+2),'supervision':'Currentfourtag answerIDs+format supervised; placeholder evidence/analysis/confidence masked','levels':dict(collections.Counter(r['native_level'] for r in chosen))}
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(count,manifest['row_sha256'],manifest['updates'])
assert selected[10]<selected[20]<selected[30]
(ROOT/'nested_subset_contract.json').write_text(json.dumps({'conditions':['fresh_base','positive10','positive20','positive30'],'same_two_negative_ids':[r['row_id'] for r in neg],'positive_subsets_nested':True,'positive20_original_preserved':True,'base_evaluation_once_per_mode':True,'evaluation':{'n':488,'modes':['OFF','ON'],'OFF_max_tokens':23000,'ON_max_tokens':10000,'temperature':1.0},'note':'Heldoutpanelusedforuserrequestedinitializationchoice; subsequentresults onthispanelareselectionexposed.'},indent=2)+'\n')
