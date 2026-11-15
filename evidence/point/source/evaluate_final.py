import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from core import *
from train_point import deterministic
from hog_pipeline import Adapter,norm
from threadpoolctl import threadpool_limits

def main():
    verify_manifest();frozen=json.loads((ROOT/'models_frozen.json').read_text());require(frozen['manifest_sha256']==sha(ROOT/'manifest.json'),'stale freeze')
    for n,h in frozen['models'].items():require(sha(ROOT/n)==h,'model changed after freeze')
    p=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'final_results';out.mkdir(exist_ok=False);deterministic(17)
    d=GlyphData('test');v=np.load(ROOT/'data/test/views.npz');kernels=np.load(ROOT/'data/test/kernels.npy');n=len(v['rows']);require(torch.cuda.is_available(),'GPU evaluation required')
    for job in p['jobs']:
        model=System().cuda();checkpoint=torch.load(ROOT/'runs'/job['name']/'final_weights.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model'],strict=True);model.eval();del checkpoint
        all_logits=[];all_stim=[];kt=torch.from_numpy(kernels).cuda()
        with torch.no_grad():
            for mi in range(5):
                logits=[];stims=[]
                for start in range(0,n,16):
                    rows=v['rows'][start:start+16];x,_,_,_=d.batch(rows,points=v['points'][start:start+16]);z,per,st,spv=model(x.cuda(),kt[mi]);require(torch.isfinite(z).all(),'finite logits');logits.append(z.cpu().numpy());stims.append(st.cpu().numpy().astype('uint8'))
                all_logits.append(np.concatenate(logits));all_stim.append(np.concatenate(stims))
        logits=np.stack(all_logits);stim=np.stack(all_stim)
        require(all(np.array_equal(stim[0],stim[i]) for i in range(5)),'encoder map leakage')
        np.savez_compressed(out/f"{job['name']}.npz",logits=logits,predictions=logits.argmax(-1),stimulation=stim,labels=d.y,fonts=d.fonts,rows=v['rows'],points=v['points'],mapping_seeds=np.array(p['test_mapping_seeds']))
        print(job['name'],'evaluation complete',flush=True);del model,kt
        torch.cuda.empty_cache()
    for kind in ['skeleton','canny']:
        adapter=Adapter(kind);bank=np.load(ROOT/f'hog_models/{kind}.npz');selected=np.stack([adapter.select(d.patch(int(r),pt)[0]) for r,pt in zip(v['rows'],v['points'])]);scores={k:[] for k in ['ideal','mma']}
        for mi in range(5):
            features=norm(np.stack([adapter.feature(sel,kernels[mi]) for sel in selected]))
            for condition in scores:scores[condition].append(features@bank[condition].T)
        for condition,ss in scores.items():
            ss=np.stack(ss);np.savez_compressed(out/f'{kind}_{condition}.npz',scores=ss,predictions=ss.argmax(-1),selected=selected,labels=d.y,fonts=d.fonts,rows=v['rows'],points=v['points'],mapping_seeds=np.array(p['test_mapping_seeds']))
        print(kind,'evaluation complete',flush=True)
    save(out/'completion.json',{'completed':True,'manifest_sha256':sha(ROOT/'manifest.json'),'model_freeze_sha256':sha(ROOT/'models_frozen.json'),'files':{f.name:sha(f) for f in out.glob('*.npz')}})
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
