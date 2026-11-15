"""Seal every model before allowing final evaluation."""
from core import *
def main():
    verify_manifest();p=json.loads((ROOT/'protocol.json').read_text());models={}
    executions={}
    for j in p['jobs']:
        folder=ROOT/'runs'/j['name'];c=json.loads((folder/'completion.json').read_text())
        require(c['completed'] and c['steps']==5000 and c['manifest_sha256']==sha(ROOT/'manifest.json'),'unfinished/stale model')
        f=folder/'final_weights.pt';require(sha(f)==c['checkpoint_sha256'],'checkpoint altered');models[f.relative_to(ROOT).as_posix()]=sha(f)
        executions[j['name']]=json.loads((folder/'execution.json').read_text())
    for seed in p['initialization_seeds']:
        a=executions[f'ideal_seed{seed}'];b=executions[f'multi_seed{seed}']
        require(a['initial_model_sha256']==b['initial_model_sha256'],'paired initialization mismatch')
        require(a['schedule_sha256']==b['schedule_sha256']==sha(ROOT/'data/train/schedule.npz'),'shared schedule mismatch')
    c=json.loads((ROOT/'hog_models/completion.json').read_text());require(c['completed'] and c['manifest_sha256']==sha(ROOT/'manifest.json'),'unfinished HOG')
    for k in ['skeleton','canny']:
        f=ROOT/f'hog_models/{k}.npz';models[f.relative_to(ROOT).as_posix()]=sha(f)
    target=ROOT/'models_frozen.json';require(not target.exists(),'freeze already exists')
    save(target,{'manifest_sha256':sha(ROOT/'manifest.json'),'models':models,'selection':'all eight artifacts, no test scores used'})
if __name__=='__main__':main()
