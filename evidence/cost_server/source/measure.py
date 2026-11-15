"""Separate latency, component and memory workers; all inputs are streamed."""
import argparse,gc,json,os,sys,time
from pathlib import Path

class NpyRows:
    """Read just one contiguous first-axis record; no mmap resident-page accumulation."""
    def __init__(self,path):
        import numpy as np
        self.f=open(path,'rb');version=np.lib.format.read_magic(self.f)
        if version==(1,0):shape,fort,dtype=np.lib.format.read_array_header_1_0(self.f)
        elif version==(2,0):shape,fort,dtype=np.lib.format.read_array_header_2_0(self.f)
        else:raise ValueError('unsupported npy version')
        if fort or dtype.hasobject:raise ValueError('require plain C-order npy')
        self.shape=shape;self.dtype=dtype;self.offset=self.f.tell();self.stride=int(np.prod(shape[1:]))*dtype.itemsize
    def get(self,i):
        import numpy as np
        if not 0<=int(i)<self.shape[0]:raise IndexError(i)
        self.f.seek(self.offset+int(i)*self.stride);b=self.f.read(self.stride)
        if len(b)!=self.stride:raise ValueError('short npy read')
        return np.frombuffer(b,dtype=self.dtype).reshape(self.shape[1:]).copy()
    def close(self):self.f.close()

def rss():
    fields={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
    return {'rss_bytes':int(fields['VmRSS'].split()[0])*1024,'hwm_bytes':int(fields['VmHWM'].split()[0])*1024}

def main():
    import resource
    ap=argparse.ArgumentParser()
    for key in ('root','artifacts','out','config','mode','cores'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--round',type=int,default=0);a=ap.parse_args()
    cores={int(x) for x in a.cores.split(',')};os.sched_setaffinity(0,cores)
    assert os.sched_getaffinity(0)==cores and len(cores)==2
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
    root=Path(a.root);art=Path(a.artifacts);out=Path(a.out);here=Path(__file__).resolve().parent
    sys.path.insert(0,str(root))
    import numpy as np
    import torch
    from torch.nn import functional as F
    from threadpoolctl import threadpool_limits,threadpool_info
    from deployment import PointInference,HogInference,patch
    from budget_adapter import render
    from hog_pipeline import norm
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ispoint=a.config.startswith('point');device='cuda' if a.config.endswith('cuda') else 'cpu'
    if device=='cuda':assert torch.cuda.is_available() and torch.cuda.device_count()==1
    protocol=json.loads((here/'measurement_protocol.json').read_text())
    with np.load(here/'evaluation_contract.npz',allow_pickle=False) as f:
        rows=f['rows'];points=f['points'];expected=f[a.config];labels=f['labels']
    schedule=np.asarray(protocol['instance_orders'][a.round if a.mode=='latency' else 0],dtype=np.int32)
    assert schedule.shape==(3600,) and np.array_equal(np.sort(schedule),np.arange(3600))
    report={'config':a.config,'mode':a.mode,'round':a.round,'affinity':sorted(cores),'device':device,'batch':1,'n':3600,
        'thread_limit':2,'interop':1,'input_buffer':'one CPU glyph and one current test mapping; no per-glyph feature cache',
        'model_file':f'point_multi_seed{a.config.split("_")[1]}.pt' if ispoint else f'skeleton_{a.config.split("_")[1]}.npy'}
    report['shared_static_assets']={'current_kernel_float32_bytes':256*96*96*4,'one_glyph_float32_bytes':96*96*4,
        'test_kernel_file_bytes':(root/'data/test/kernels.npy').stat().st_size,
        'evaluation_contract_file_bytes':(here/'evaluation_contract.npz').stat().st_size,
        'note':'Kernel/input file reads and mapping transfer occur outside request latency; process RSS includes transient streaming buffers.'}
    with threadpool_limits(limits=2):
        report['threadpools']=threadpool_info();assert all(x['num_threads']==2 for x in report['threadpools'])
        report['memory_before_model']=rss()
        if ispoint:
            model=PointInference().eval().requires_grad_(False)
            state=torch.load(art/report['model_file'],map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True);del state;gc.collect();model.to(device)
            assert not any('teacher' in n for n,_ in model.named_modules())
            report['model_state_bytes']=sum(t.numel()*t.element_size() for t in model.state_dict().values())
            report['model_parameters']=sum(p.numel() for p in model.parameters())
        else:
            model=HogInference(art/report['model_file']);report['model_state_bytes']=model.bank.nbytes
            # Record implementation-specific static NumPy buffers without counting aliases twice.
            seen=set();buffers=[]
            def inventory(obj,path,depth=0):
                if id(obj) in seen:return
                seen.add(id(obj))
                if isinstance(obj,np.ndarray):buffers.append({'name':path,'shape':list(obj.shape),'dtype':str(obj.dtype),'bytes':obj.nbytes});return
                if depth<5 and hasattr(obj,'__dict__'):
                    for k,v in vars(obj).items():inventory(v,path+'.'+k,depth+1)
                elif depth<5 and isinstance(obj,(tuple,list)):
                    for i,v in enumerate(obj):inventory(v,path+'.'+str(i),depth+1)
            inventory(model.adapter,'adapter');report['algorithm_static_array_inventory']=buffers
        report['model_file_bytes']=(art/report['model_file']).stat().st_size
        report['memory_after_model']=rss()
        sync=(lambda:torch.cuda.synchronize()) if device=='cuda' else (lambda:None)
        # Training-only warmup: fixed glyph and mapping, discarded before test requests.
        warm=protocol['warmup'];wg=NpyRows(root/'data/train/glyphs.npy');wk=NpyRows(root/'data/train/kernels.npy')
        glyph=wg.get(warm['row']);kernel_np=wk.get(0);kernel=torch.from_numpy(kernel_np).to(device) if ispoint else kernel_np
        count=0;start=time.perf_counter()
        while count<20 or time.perf_counter()-start<2.:
            model.predict(glyph,warm['point'],kernel);sync();count+=1
        report['warmup']={'requests':count,'elapsed_seconds':time.perf_counter()-start,'row':warm['row'],'point':warm['point']}
        wg.close();wk.close();del wg,wk,glyph,kernel,kernel_np;gc.collect()
        if device=='cuda':
            report['cuda_peak_before_measurement']={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        glyphs=NpyRows(root/'data/test/glyphs.npy');kernels=NpyRows(root/'data/test/kernels.npy')
        assert glyphs.shape==(720,96,96) and kernels.shape==(5,256,96,96)
        current=-1;kernel=None;kernel_np=None
        def prepare(mi):
            nonlocal current,kernel,kernel_np
            if mi!=current:
                # Release old current kernel before creating its replacement.
                kernel=None;kernel_np=None
                kernel_np=kernels.get(mi)
                kernel=torch.from_numpy(kernel_np).to(device) if ispoint else kernel_np
                current=mi
            sync()
        prepare(int(schedule[0])//720)
        report['memory_after_warmup_current_kernel']=rss()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        predictions=np.empty(3600,dtype=np.int64);times=np.empty(3600,dtype=np.int64) if a.mode=='latency' else None
        components=np.empty((3600,3),dtype=np.int64) if a.mode=='components' else None
        memory_samples=[]
        for j,flat in enumerate(schedule):
            mi,i=divmod(int(flat),720);prepare(mi);glyph=glyphs.get(int(rows[i]));pt=points[i]
            if a.mode=='latency':
                start=time.perf_counter_ns();prediction=model.predict(glyph,pt,kernel);end=time.perf_counter_ns()
                times[j]=end-start
            elif a.mode=='components':
                # Separate diagnostic process: synchronization overhead does not contaminate primary latency.
                with torch.inference_mode():
                    sync();t0=time.perf_counter_ns()
                    if ispoint:
                        x=patch(glyph,pt)[None].to(device);selected=model.encoder(x)
                    else:selected=model.adapter.select(patch(glyph,pt))
                    sync();t1=time.perf_counter_ns()
                    if ispoint:spv=render(selected,kernel)
                    else:spv=kernel[selected].sum(axis=0,dtype=np.float32)
                    sync();t2=time.perf_counter_ns()
                    if ispoint:
                        vision=F.interpolate(spv,size=(128,128),mode='bilinear',align_corners=False)
                        logits=model.classifier(model.blind_feat(model.channel(vision)));prediction=int(logits.argmax(1).cpu().item())
                    else:prediction=int((norm(model.adapter.hog.extract(spv))@model.bank.T).argmax())
                    sync();t3=time.perf_counter_ns()
                components[j]=(t1-t0,t2-t1,t3-t2)
                del selected,spv
                if ispoint:del x,vision,logits
            else:
                prediction=model.predict(glyph,pt,kernel);sync()
                if j%120==0:memory_samples.append({'completed':j+1,**rss()})
            predictions[j]=prediction
            del glyph
            if (j+1)%360==0:print(a.config,a.mode,a.round,j+1,'/3600',flush=True)
        target=expected.reshape(-1)[schedule];bad=np.flatnonzero(predictions!=target)
        report['prediction_mismatches']=[{'flat':int(schedule[k]),'expected':int(target[k]),'got':int(predictions[k])} for k in bad]
        report['status']='PASS' if len(bad)==0 else 'REVIEW_PREDICTION_DRIFT'
        report['accuracy']=float(np.mean(predictions==labels[rows][schedule%720]))
        report['memory_at_end']=rss();report['process_lifetime_peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        if device=='cuda':report['cuda_measurement_peak']={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        payload={'order':schedule,'predictions':predictions}
        if times is not None:
            assert np.all(times>0);payload['latency_ns']=times
            report['median_ms']=float(np.median(times)/1e6);report['p95_ms']=float(np.percentile(times,95)/1e6)
        if components is not None:
            payload['component_ns']=components
            report['component_medians_ms']=(np.median(components,axis=0)/1e6).tolist()
            report['without_render_median_ms']=float(np.median(components[:,0]+components[:,2])/1e6)
        if a.mode=='memory':report['memory_samples']=memory_samples
        glyphs.close();kernels.close()
        report['final_affinity']=sorted(os.sched_getaffinity(0));assert report['final_affinity']==sorted(cores)
        stem=f'{a.mode}_r{a.round}_{a.config}'
        np.savez_compressed(out/(stem+'.npz'),**payload)
        (out/(stem+'.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
        print(report['status'],flush=True)
    if report['status']!='PASS':raise RuntimeError('prediction drift: retain entire run, do not selectively repeat')
if __name__=='__main__':main()
