"""Resume only after development has returned successfully."""
import subprocess,sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    assert (ROOT/'selected_config.json').exists(), 'development not complete'
    for script in ['prepare_external.py','evaluate.py','summarize.py','measure_costs.py']:
        print('PIPELINE',script,flush=True)
        completed=subprocess.run([sys.executable,'-B',str(ROOT/script)])
        if completed.returncode:
            (ROOT/'pipeline_failure.json').write_text(json.dumps(dict(script=script,exit_code=completed.returncode,time=time.time())),encoding='utf-8')
            raise SystemExit(completed.returncode)
    print('PIPELINE COMPLETE',flush=True)
