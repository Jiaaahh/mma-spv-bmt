import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc
from core import *
from train_point import deterministic
from hog_pipeline import Adapter,event_counts,norm
from threadpoolctl import threadpool_limits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');a=ap.parse_args();device='cuda' if a.cuda else 'cpu'
    verify_manifest();require(torch.__version__.split('+')[0]=='2.6.0' and np.__version__=='1.26.4','pinned torch/numpy');deterministic(17)
    from importlib.metadata import version
    require(version('scikit-image')=='0.25.0' and version('scipy')=='1.13.1','pinned HOG packages')
    d=GlyphData();s=np.load(ROOT/'data/train/schedule.npz');require(s['rows'].shape==(5000,16) and s['points'].shape==(5000,16,2),'schedule shape');require(s['rows'].min()>=0 and s['rows'].max()<1080,'schedule training isolation')
    require(np.array_equal(s['mapping_index'],np.arange(5000)%5),'mapping schedule');p=d.p
    rng=np.random.default_rng(p['schedule_row_seed']);rr=np.stack([rng.choice(1080,16,replace=False) for _ in range(5000)]);require(np.array_equal(rr,s['rows']),'row replay')
    rng=np.random.default_rng(p['schedule_point_seed']);pts=np.stack([np.stack([d.point(int(r),rng) for r in rows]) for rows in rr]);require(np.array_equal(pts,s['points']),'point replay')
    events=event_counts(s);require(sum(int(v.sum()) for v in events.values())==80000,'deduplication multiplicities')
    ref=np.load(ROOT/'data/train/patch_reference.npz');x,y,_,_=d.batch(ref['rows'],points=ref['points']);delta=float(np.abs(x.numpy()-ref['patches']).max());require(np.allclose(x.numpy(),ref['patches'],atol=2e-6,rtol=1e-5),'bilinear tolerance')
    k=np.load(ROOT/'data/train/kernels.npy');require(k.shape==(5,256,96,96) and np.allclose(k.sum((2,3),dtype=np.float64),1,atol=1e-6),'L1 kernels')
    kt=torch.from_numpy(k).to(device);errors=[]
    for kind in ['skeleton','canny']:
        adapter=Adapter(kind);sel=adapter.select(x[0]);require(len(np.unique(sel))==64,'HOG fixed budget');stim=torch.zeros(1,256,device=device).scatter_(1,torch.tensor(sel[None],device=device),1.)
        for mi in range(5):
            actual=render(stim,kt[mi])[0,0].cpu().numpy();reference=k[mi,sel].sum(0,dtype=np.float32);errors.append(float(np.abs(actual-reference).max()));require(np.allclose(actual,reference,atol=2e-6,rtol=1e-5),'renderer parity')
            feat=adapter.feature(sel,k[mi]);require(feat.shape==(4356,) and np.isfinite(feat).all(),'HOG feature')
    model=System().to(device).train();before=state_hash(model.teacher);wbefore=model.encoder.seq1[0].weight.detach().clone();opt=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=1e-4)
    for mi in [0,1]:
        opt.zero_grad(set_to_none=True);output=model(x[:2].to(device),kt[mi]);loss,ce,per=loss_parts(output,y[:2].to(device));require(torch.isfinite(loss),'loss finite');loss.backward();require(model.encoder.seq1[0].weight.grad.norm()>0,'encoder gradient');opt.step()
    require(not torch.equal(wbefore,model.encoder.seq1[0].weight),'weight update');require(state_hash(model.teacher)==before,'teacher frozen')
    model.eval()
    with torch.no_grad():
        one=model(x[:2].to(device),kt[0]);two=model(x[:2].to(device),kt[1]);require(torch.equal(one[2],two[2]),'map-independent encoder');require(((one[2]==0)|(one[2]==1)).all() and (one[2].sum(1)==64).all(),'hard64')
    report={'passed':True,'device':device,'manifest_sha256':sha(ROOT/'manifest.json'),'schedule_replayed':True,'patch_max_abs_error':delta,'render_max_abs_errors':errors,'two_full_random_weight_resnet152_updates':True,'test_scores_computed':False};save(ROOT/f'engineering/contracts_{device}.json',report);print(json.dumps(report))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
