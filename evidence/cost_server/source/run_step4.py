import argparse,datetime,hashlib,importlib.metadata,json,os,platform,subprocess,sys,traceback,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def check(root,assets):
    result={}
    for n,v in assets.items():
        p=(root/n).resolve()
        if not p.is_relative_to(root):raise RuntimeError('asset escape')
        if not p.exists() and not v.get('required',True):result[n]='ABSENT_OPTIONAL';continue
        result[n]=sha(p)
        if result[n]!=v['sha256']:raise RuntimeError('SHA mismatch '+str(p))
    return result
def select_cores(topology):
    groups={}
    for entry in sorted(topology,key=lambda x:x['cpu']):
        groups.setdefault((entry['socket'],entry['numa']),{}).setdefault(entry['core'],entry['cpu'])
    for group,cores in sorted(groups.items()):
        if len(cores)>=2:return sorted(cores.values())[:2]
    raise RuntimeError('Cannot identify two distinct physical cores on same NUMA/socket; no fallback to arbitrary logical CPUs')
def choose_cores():
    topology=[]
    for cpu in sorted(os.sched_getaffinity(0)):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}')
        socket=int((p/'topology/physical_package_id').read_text());core=int((p/'topology/core_id').read_text())
        nodes=sorted(int(n.name[4:]) for n in p.glob('node[0-9]*'))
        if len(nodes)!=1:continue
        entry={'cpu':cpu,'socket':socket,'core':core,'numa':nodes[0]};topology.append(entry)
    return select_cores(topology),topology
def gpu():
    return subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
def idle_gpu():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if raw:raise RuntimeError('GPU compute processes present; do not kill automatically: '+raw)
def snapshot():
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':gpu()}
    for name,p in [('cpu_quota','/sys/fs/cgroup/cpu.max'),('memory_limit','/sys/fs/cgroup/memory.max'),('cpu_stat','/sys/fs/cgroup/cpu.stat'),('loadavg','/proc/loadavg'),('boot_id','/proc/sys/kernel/random/boot_id')]:
        path=Path(p);result[name]=path.read_text().strip() if path.exists() else None
    result['gpu_telemetry']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.current.sm,power.draw','--format=csv,noheader'],text=True).strip()
    return result
def summarize(reports,configs):
    import numpy as np
    summary={'latency':{},'memory':{},'components':{},'cpu_point_over_mma_ratios':{}}
    for config in configs:
        rounds=[]
        for r in range(3):
            name=f'latency_r{r}_{config}'
            info=json.loads((reports/(name+'.json')).read_text())
            with np.load(reports/(name+'.npz')) as arr:
                assert len(arr['latency_ns'])==3600 and np.all(arr['latency_ns']>0)
                median=float(np.median(arr['latency_ns'])/1e6);p95=float(np.percentile(arr['latency_ns'],95)/1e6)
            assert median==info['median_ms'] and p95==info['p95_ms'] and info['status']=='PASS'
            rounds.append({'round':r,'median_ms':median,'p95_ms':p95})
        medians=[x['median_ms'] for x in rounds]
        summary['latency'][config]={'rounds':rounds,'median_of_round_medians_ms':float(np.median(medians)),'round_median_range_ms':[min(medians),max(medians)]}
        m=json.loads((reports/f'memory_r0_{config}.json').read_text());c=json.loads((reports/f'components_r0_{config}.json').read_text())
        summary['memory'][config]={k:v for k,v in m.items() if k.startswith(('memory_','process_lifetime','cuda_','model_','algorithm_static'))}
        summary['components'][config]={k:c[k] for k in ('component_medians_ms','without_render_median_ms')}
    for seed in (17,29,43):
        values=[summary['latency'][f'point_{seed}_cpu']['rounds'][r]['median_ms']/summary['latency']['skeleton_mma_cpu']['rounds'][r]['median_ms'] for r in range(3)]
        summary['cpu_point_over_mma_ratios'][str(seed)]={'rounds':values,'median':float(np.median(values)),'range':[min(values),max(values)]}
    return summary
def main():
    ap=argparse.ArgumentParser()
    for k in ('root','artifacts','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();root=Path(a.root).resolve();art=Path(a.artifacts).resolve();out=Path(a.out).resolve()
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in (root,art,HERE)):raise RuntimeError('Use a fresh output directory outside inputs')
    for n,h in json.loads((HERE/'PACKAGE_MANIFEST.json').read_text(encoding='utf8')).items():
        if sha(HERE/n)!=h:raise RuntimeError('execution package changed: '+n)
    expected=json.loads((HERE/'expected_assets.json').read_text());exports=json.loads((HERE/'accepted_artifacts.json').read_text())
    protocol=json.loads((HERE/'measurement_protocol.json').read_text())
    out.mkdir(parents=True);reports=out/'reports';reports.mkdir()
    report={'stage':4,'status':'FAIL','root':str(root),'artifacts':str(art),'package_manifest_sha256':sha(HERE/'PACKAGE_MANIFEST.json'),
        'protocol_sha256':sha(HERE/'measurement_protocol.json'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]}
    try:
        if platform.system()!='Linux':raise RuntimeError('Linux server required')
        report['assets_before']=check(root,expected['assets']);report['exports_before']=check(art,exports)
        report['python']=platform.python_version();report['versions']={k:importlib.metadata.version(k) for k in expected['versions']}
        assert report['python']==expected['python_version'] and report['versions']==expected['versions'],'environment drift'
        idle_gpu();initial=snapshot();report['initial_environment']=initial
        if len(initial['gpu'].splitlines())!=1:raise RuntimeError('single GPU required')
        report['platform']=platform.platform();report['lscpu']=subprocess.check_output(['lscpu'],text=True)
        cores,topology=choose_cores();report['selected_cpus']=cores;report['allowed_cpu_topology']=topology
        quota=initial['cpu_quota']
        if quota and not quota.startswith('max'):
            q,p=map(int,quota.split());assert q/p>=2,'CPU quota below two threads'
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='2'
        for number,job in enumerate(protocol['jobs']):
            idle_gpu();before=snapshot()
            assert before['gpu']==initial['gpu'] and before['boot_id']==initial['boot_id'],'host changed'
            assert before['cpu_quota']==initial['cpu_quota'] and before['memory_limit']==initial['memory_limit'],'resource limits changed'
            config=job['config'];mode=job['mode'];r=job['round'];stem=f'{mode}_r{r}_{config}'
            print(f'STEP4 {number+1}/{len(protocol["jobs"])} {stem}',flush=True)
            cmd=[sys.executable,'-B',str(HERE/'measure.py'),'--root',str(root),'--artifacts',str(art),'--out',str(reports),'--config',config,'--mode',mode,'--round',str(r),'--cores',','.join(map(str,cores))]
            with (reports/(stem+'.log')).open('w',encoding='utf8') as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
            after=snapshot();entry={**job,'before':before,'after':after,'returncode':proc.returncode};report['jobs'].append(entry)
            assert after['gpu']==initial['gpu'] and after['boot_id']==initial['boot_id'],'host changed'
            if proc.returncode:raise RuntimeError('worker failed: '+stem+'; retain all results, no selective rerun')
            idle_gpu()
        summary=summarize(reports,protocol['configs'])
        (reports/'cost_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        report['status']='COMPLETE_PENDING_LOCAL_REVIEW'
    except Exception:report['error']=traceback.format_exc()
    finally:
        try:
            report['assets_after']=check(root,expected['assets']);report['exports_after']=check(art,exports)
            assert report['assets_after']==report.get('assets_before') and report['exports_after']==report.get('exports_before')
        except Exception:report['postcheck_error']=traceback.format_exc();report['status']='FAIL'
        report['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (reports/'step4_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        manifest={p.name:sha(p) for p in sorted(reports.iterdir())}
        (reports/'RETURN_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
        dest=out/('SEND_mma_cost_step4_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip')
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(reports.iterdir()):z.write(p,'reports/'+p.name)
        dest.with_suffix('.zip.sha256.txt').write_text(sha(dest)+'  '+dest.name+'\n')
        print(report['status']);print('SEND:',dest)
    return 0 if report['status']=='COMPLETE_PENDING_LOCAL_REVIEW' else 1
if __name__=='__main__':sys.exit(main())
