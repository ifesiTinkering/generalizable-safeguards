import json,sys,re,collections
from pathlib import Path
P=Path('/workspace/claim2-evidence-ablations-20261111/claim2_format22_pilot_20261112');R=P.parent/'runtime_v9_tinker_template';O=Path('/workspace/safepyramid-runs/claim2_evidence_ablations_20261111/partial/pilot_sft_learningcurve_20261112');sys.path[:0]=[str(P),str(R)]
from reward_pipeline import parse
rows={r['row_id']:r for r in json.loads((P/'validation_rows.json').read_text())};out={}
for mode in ['OFF','ON']:
 files=sorted((O/'evaluations'/mode/'format32/raw').glob('*.json'));counts=collections.Counter();examples={};answercounts=collections.Counter();details=collections.Counter()
 for f in files:
  r=json.loads(f.read_text());text=r['text'];row=rows[r['row_id']];allowed=[x['rule_id']for x in row['all_policy_rules']];a=re.search(r'<answer>(.*?)</answer>',text,re.S)
  try:
   av=json.loads(a.group(1)) if a else None;answerok=isinstance(av,list)and all(type(x)is str for x in av)and len(set(av))==len(av)and set(av)<=set(allowed)
  except Exception:answerok=False
  try:parse(text,allowed);reason='valid'
  except Exception as e:reason=str(e) if type(e)is ValueError else type(e).__name__
  if not r['clean_termination']:
   details['unclean_but_parser_'+reason]+=1;reason='unclean_termination'
  counts[reason]+=1
  if reason=='answer_ids' and a:
   av=json.loads(a.group(1));details['answer_duplicate_ids' if all(type(z)is str for z in av) and len(av)!=len(set(av)) else 'answer_unknown_or_wrong_type']+=1
  if reason=='JSONDecodeError':
   try:json.loads(a.group(1));details['confidence_JSON_error']+=1
   except Exception:details['answer_JSON_error']+=1
  if reason=='quote_attributes':
   import xml.etree.ElementTree as ET
   ev=re.search(r'<evidence>(.*?)</evidence>',text,re.S);root=ET.fromstring('<evidence>'+ev.group(1)+'</evidence>')
   bad_speakers={q.attrib.get('speaker') for q in root.iter('quote') if q.attrib.get('speaker')not in ['assistant','user']}
   details['quote_failure_rows_chatbot_only' if bad_speakers=={'chatbot'} else 'quote_failure_rows_other_speakers']+=1
   for q in root.iter('quote'):
    attrs=q.attrib
    if set(attrs)!={'turn_id','speaker'} or not re.fullmatch(r'[0-9]+',attrs.get('turn_id','')) or attrs.get('speaker')not in ['assistant','user']:
     details['bad_quote_attributes:'+json.dumps(attrs,sort_keys=True)]+=1

  if reason!='valid':
   answercounts['valid_answer_contract' if answerok else 'invalid_answer_contract']+=1
   for tag in ['answer','evidence','analysis','confidence']:
    if f'<{tag}>'not in text:details['missing_open_'+tag]+=1
    if f'</{tag}>'not in text:details['missing_close_'+tag]+=1
   if len(r['tokens'])>=r['sampling_config']['max_tokens']:details['at_token_cap']+=1
   if reason=='unreadable_evidence':
    import xml.etree.ElementTree as ET
    ev=re.search(r'<evidence>(.*?)</evidence>',text,re.S)
    try:ET.fromstring('<evidence>'+ev.group(1)+'</evidence>')
    except ET.ParseError as e:details['xml_'+str(e).split(': line')[0]]+=1
   if len(examples.setdefault(reason,[]))<3:examples[reason].append({'row_id':r['row_id'],'token_count':len(r['tokens']),'clean':r['clean_termination'],'answer_contract_valid':answerok,'tags':re.findall(r'<[^>]{0,120}>',text)[:45],'head':text[:180],'tail':text[-220:]})
 out[mode]={'available_rows':len(files),'not_final':len(files)!=488,'reasons':dict(counts),'malformed_answer_contract':dict(answercounts),'malformed_details':dict(details),'examples':examples}
print(json.dumps(out))
