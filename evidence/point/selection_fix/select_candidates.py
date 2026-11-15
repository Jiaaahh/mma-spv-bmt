from pathlib import Path
import hashlib,json
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--results-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
P=args.results_root.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(candidates):return sorted(candidates,key=lambda v:(-v['correct'],round(v['ce'],8),v['name']))
def main():
 config=json.loads((P/'selection_protocol.json').read_text());results=[];initial=[];schedule=[]
 for job in config['jobs']:
  d=P/job['name'];run=d/'runs/point_window';completion=json.loads((run/'completion.json').read_text());ex=json.loads((run/'execution.json').read_text())
  assert completion['completed'] and completion['manifest_sha256']==sha(d/'manifest.json')
  log=[json.loads(x) for x in (run/'training.jsonl').read_text().splitlines()];assert [x['step'] for x in log]==list(range(1,5001))
  assert all(x['mapping_index']==(0 if job['condition']=='ideal' else (x['step']-1)%5) for x in log)
  with np.load(run/'inner_5000.npz') as a:
   logits=a['logits'].astype(np.float64);labels=a['labels'];assert logits.shape==(900,180) and np.isfinite(logits).all();correct=int((logits.argmax(1)==labels).sum());s=logits-logits.max(1,keepdims=True);ce=float((np.log(np.exp(s).sum(1))-s[np.arange(900),labels]).mean())
  reported=json.loads((run/'metrics_5000.json').read_text())['inner'];assert correct==reported['correct'] and abs(ce-reported['ce'])<1e-10
  initial.append(ex['initial_model_sha256']);schedule.append(ex['schedule_sha256']);results.append({**job,'correct':correct,'n':900,'ce':ce,'checkpoint_sha256':completion['checkpoint_sha256']})
 assert len(set(initial))==len(set(schedule))==1,'initialization or schedule mismatch'
 chosen={c:rank([v for v in results if v['condition']==c])[0] for c in ['ideal','multi']}
 args.output.write_text(json.dumps({'all_candidates':results,'selected':chosen,'status':'development_selection_only','protocol_sha256':sha(P/'selection_protocol.json')},indent=2));print(json.dumps(chosen,indent=2))
if __name__=='__main__':main()
