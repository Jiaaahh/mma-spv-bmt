from pathlib import Path
from collections import defaultdict
import hashlib,json,csv
R=Path(__file__).resolve().parent
manifest=json.loads((R/'manifest.json').read_text(encoding='utf8'))
report={'hash_checks':[],'csv_counts':{},'scope':'Frozen prediction and recorded-cost verification; no model replay.'}
for r in manifest:
 p=R/r['path'];actual=hashlib.sha256(p.read_bytes()).hexdigest();ok=actual==r['sha256']
 report['hash_checks'].append({'path':r['path'],'matched':ok})
 if not ok:raise RuntimeError('Hash mismatch: '+r['path'])
 if p.name=='predictions.csv' or p.name=='merged_predictions_with_truth.csv':
  rows=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
  if not rows:continue
  fields=rows[0]
  pair=next(((a,b)for a,b in [('y_true','y_pred'),('true_label','predicted_label'),('class_id','predicted_class_id'),('class_id','predicted'),('class_id','cnn_prediction')]if a in fields and b in fields),None)
  flag=next((k for k in ['correct','cnn_correct']if k in fields),None)
  if pair is None and flag is None:continue
  groups=defaultdict(lambda:[0,0]);mismatch=0
  for row in rows:
   correct=(row[pair[0]]==row[pair[1]])if pair else int(float(row[flag]))
   if pair and flag:mismatch+=int(correct!=int(float(row[flag])))
   key='|'.join(str(row.get(k,''))for k in ['panel','method','condition','budget','projection_seed'])
   groups[key][0]+=int(correct);groups[key][1]+=1
  report['csv_counts'][r['path']]={'rows':len(rows),'flag_disagreements':mismatch,'groups':{k:{'correct':v[0],'n':v[1],'accuracy_percent':100*v[0]/v[1]}for k,v in groups.items()}}
  if mismatch:raise RuntimeError('Correctness flag mismatch '+r['path'])
try:
 import numpy as np
 cost=json.loads((R/'randumb/costs.json').read_text(encoding='utf8'));rr={}
 for name,obj in cost['methods'].items():
  rr[name]={}
  for batch,vals in obj.get('inference',{}).items():
   raw=vals.get('raw_seconds')
   if raw:rr[name][batch]={'n':len(raw),'median_ms':float(np.median(raw)*1000),'p95_ms':float(np.quantile(raw,.95)*1000)}
 report['randumb_raw_cost']=rr
 report['randumb_mma_time_ratio']={batch:rr['R1'][batch]['median_ms']/rr['H1'][batch]['median_ms'] for batch in ['1','32']}
 p=R/'whole_glyph/StudyB/merged_predictions_with_truth.csv'
 if p.exists():
  rows=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')));out={}
  for method in ['ideal_only','ideal_plus_multi4']:
   acc=[np.mean([r['y_true']==r['y_pred']for r in rows if r['method']==method and int(r['budget'])==b])for b in [32,48,64,96,128]]
   out[method]={'accuracy':acc,'naubc':float(np.dot(acc,np.array([8,16,24,32,16])/96))}
  out['delta_naubc']=out['ideal_plus_multi4']['naubc']-out['ideal_only']['naubc'];report['budget']=out
except ImportError:report['optional_numpy']='not installed; numerical array checks skipped'
(R/'verification_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'hashes_passed':len(report['hash_checks']),'csv_files_recomputed':len(report['csv_counts']),'budget':report.get('budget'),'randumb_cost':report.get('randumb_raw_cost')},ensure_ascii=False))
