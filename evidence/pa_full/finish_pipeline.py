"""Await complete verified features, then evaluate, independently audit and report."""
from pathlib import Path
import time,json,subprocess,sys
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
start=time.monotonic()
while True:
    p=OUT/'progress.json'
    if p.exists():
        data=json.loads(p.read_text(encoding='utf-8'))
        if data.get('stage')=='FEATURE_EXTRACTION_COMPLETE':break
    if time.monotonic()-start>7200:raise TimeoutError('No completed feature manifest after two hours; scoring not started')
    time.sleep(10)
for name in ['evaluate.py','independent_audit.py','make_report.py']:
    print('START',name,flush=True)
    subprocess.run([sys.executable,str(OUT/name)],cwd=ROOT,check=True)
    print('COMPLETE',name,flush=True)
print('PIPELINE COMPLETE',flush=True)
