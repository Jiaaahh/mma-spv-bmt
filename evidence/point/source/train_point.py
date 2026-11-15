import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,time,platform
from core import *

WEIGHT_SHA='394f9c45966e3651a89bbb78a48410a6755854ce4a5ab64927cf1c7247f85e58'
def deterministic(seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--weights',type=Path,required=True);a=ap.parse_args()
    verify_manifest();p=json.loads((ROOT/'protocol.json').read_text());job=next(j for j in p['jobs'] if j['name']==a.job)
    require(torch.cuda.is_available() and torch.__version__.split('+')[0]=='2.6.0','CUDA torch2.6 required');require(sha(a.weights)==WEIGHT_SHA,'ImageNet weight hash')
    gate=json.loads((ROOT/'engineering/contracts_cuda.json').read_text());require(gate['passed'] and gate['manifest_sha256']==sha(ROOT/'manifest.json'),'CUDA gate')
    out=ROOT/'runs'/a.job;out.mkdir(parents=True,exist_ok=False);deterministic(job['seed']);start=time.time()
    d=GlyphData();s=np.load(ROOT/'data/train/schedule.npz');k=torch.from_numpy(np.load(ROOT/'data/train/kernels.npy')).cuda()
    model=System(a.weights).cuda().train();teacher=state_hash(model.teacher);optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=p['learning_rate'])
    save(out/'execution.json',{'job':job,'manifest_sha256':sha(ROOT/'manifest.json'),'schedule_sha256':sha(ROOT/'data/train/schedule.npz'),'initial_model_sha256':state_hash(model),'teacher_sha256':teacher,'weights_sha256':sha(a.weights),'python':platform.python_version(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'time_started':start})
    with (out/'training.jsonl').open('w') as log:
        for step in range(p['steps']):
            mi=0 if job['condition']=='ideal' else int(s['mapping_index'][step]);x,y,_,ink=d.batch(s['rows'][step],points=s['points'][step]);x=x.cuda();y=y.cuda()
            optimizer.zero_grad(set_to_none=True);output=model(x,k[mi]);loss,ce,per=loss_parts(output,y);require(torch.isfinite(loss),'finite loss');loss.backward();grads={}
            for name,mod in [('encoder',model.encoder),('blind',model.blind_feat),('classifier',model.classifier)]:
                g=torch.sqrt(sum((v.grad.square().sum() for v in mod.parameters() if v.grad is not None),torch.zeros((),device='cuda')));require(torch.isfinite(g) and g>0,'gradient '+name);grads[name]=float(g)
            optimizer.step();record={'step':step+1,'mapping_index':mi,'loss':float(loss.detach()),'ce':float(ce.detach()),'perceptual':float(per.detach()),'accuracy':float((output[0].argmax(1)==y).float().mean()),'gradient_norms':grads,'elapsed_seconds':time.time()-start};log.write(json.dumps(record)+'\n');log.flush()
            if (step+1)%100==0:print(json.dumps(record),flush=True)
    require(state_hash(model.teacher)==teacher,'teacher unchanged');torch.save({'model':model.state_dict(),'step':p['steps'],'protocol':p,'job':job},out/'final_weights.pt')
    save(out/'completion.json',{'completed':True,'steps':p['steps'],'job':job,'manifest_sha256':sha(ROOT/'manifest.json'),'checkpoint_sha256':sha(out/'final_weights.pt'),'teacher_final_sha256':state_hash(model.teacher),'elapsed_seconds':time.time()-start,'test_accessed':False})
if __name__=='__main__':main()
