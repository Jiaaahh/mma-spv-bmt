"""Independent read-only prediction audit; no source model scripts imported/executed."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib,sys,platform,os
from contextlib import nullcontext
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import numpy as np

B=Path('/historical/user/Documents/Codex/2026-09-22/thread-01a03819-f724-7393-a6a0-29efdf40b24e-2/work/revision_20261004/evidence_bundle')
O=Path(__file__).resolve().parent
weights_cache={}
R={'scope':'Prediction-level recalculation; no training or inference replay','environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},'sources':{},'integrity':{},'metrics':{},'comparisons':{},'checks':[]}
def readj(p):return json.loads((B/p).read_text(encoding='utf8'))
def record(p):
 p=Path(p); R['sources'][str(p)]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def chk(name,got,want,tol=1e-9):
 ok=bool(np.allclose(got,want,atol=tol,rtol=0));R['checks'].append(dict(name=name,passed=ok,actual=got,expected=want))
def f1(t,p):
 cm=np.bincount(t*180+p,minlength=180**2).reshape(180,180);den=cm.sum(0)+cm.sum(1)
 return float(np.divide(2*np.diag(cm),den,out=np.zeros(180),where=den>0).mean())
def readpred(tag,path):
 p=B/path;record(p)
 with p.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
 groups=defaultdict(list);keys=set();dups=0;flagbad=0
 for x in rr:
  y=int(next(x[k] for k in ['class_id','true_label','y_true'] if k in x)); pred=int(next(x[k] for k in ['predicted_class_id','predicted_label','y_pred','predicted_class','predicted'] if k in x))
  assert 0<=y<180 and 0<=pred<180
  panel=x.get('panel','');method=x['method'];cond=x.get('condition','');budget=x.get('budget','');rs=x.get('projection_seed','fixed');font=x.get('font_id',x.get('font'));mp=x.get('mapping_seed',x.get('seed'))
  g=(panel,method,cond,budget,rs);key=g+(font,mp,y)
  dups+=key in keys;keys.add(key);flagbad+=('correct'in x and int(x['correct'])!=int(y==pred))
  groups[g].append((y,pred,font,mp))
 ans={};R['metrics'][tag]={}
 for g,rows in groups.items():
  rows.sort(key=lambda x:(x[0],x[2],int(x[3])))
  t=np.array([x[0] for x in rows]);pred=np.array([x[1] for x in rows]);font=np.array([x[2] for x in rows]);mp=np.array([x[3] for x in rows]);cor=(t==pred).astype(float)
  fs=sorted(set(font));ms=sorted(set(mp),key=int)
  assert len(rows)==180*len(fs)*len(ms)
  assert all(np.count_nonzero((font==f)&(mp==m))==180 for f in fs for m in ms)
  cells=[(mp==m)&(font==f) for m in ms for f in fs] if g[0]=='DEV' else [mp==m for m in ms]
  v=dict(n=len(rows),correct=int(cor.sum()),accuracy_percent=float(cor.mean()*100),macro_f1_subpanel=float(np.mean([f1(t[c],pred[c]) for c in cells])),macro_f1_pooled=f1(t,pred),fonts=fs,maps=ms)
  R['metrics'][tag]['|'.join(g)]=v
  ans[g]=dict(t=t,p=pred,font=font,mp=mp,cor=cor,char=cor.reshape(180,len(fs),len(ms)).mean((1,2)),cube=cor.reshape(180,len(fs),len(ms)))
 R['integrity'][tag]=dict(rows=len(rr),groups=len(groups),unique_keys=len(keys),duplicates=dups,correct_flag_disagreements=flagbad,all_groups_complete_cartesian=True)
 assert dups==0 and flagbad==0
 print('Read '+tag+': '+str(len(rr))+' rows',flush=True)
 return ans
def boot(vals,seed,kind='multinomial'):
 vals=np.asarray(vals);rng=np.random.default_rng(seed);res=[]
 if kind=='multinomial':
  if seed not in weights_cache:weights_cache[seed]=rng.multinomial(180,np.full(180,1/180),size=100000).astype(np.int16)
  return np.quantile(weights_cache[seed]@vals/180,[.025,.975],axis=0)
 for _ in range(50):
  if kind=='multinomial':z=rng.multinomial(180,np.full(180,1/180),size=2000)@vals/180
  else:z=vals[rng.integers(0,180,size=(2000,180))].mean(1)
  res.append(z)
 return np.quantile(np.concatenate(res),[.025,.975],axis=0)
def compare(name,a,b,seed,kind='multinomial',expected=None):
 d=(a-b)*100;ci=boot(d,seed,kind).tolist();q={'difference_pp':float(d.mean()),'ci95_pp':ci,'replicates':100000,'seed':seed,'sampler':kind};R['comparisons'][name]=q
 if expected is not None:chk(name+'/CI',ci,expected)
 print(name+' '+str(ci),flush=True)
 return q
def base(groups,method,panel='',cond='',budget='',seed='fixed'):return groups[(panel,method,cond,str(budget),str(seed))]
def meanchar(groups,panel,method):return np.mean([v['char'] for k,v in groups.items() if k[0]==panel and k[1]==method],0)

with nullcontext():
 C=readpred('C1','whole_glyph/C1/predictions.csv');E=readpred('E1','whole_glyph/E1/predictions.csv');Q=readpred('StudyB','whole_glyph/StudyB/merged_predictions_with_truth.csv')
 I=readpred('internal','internal_hog/results/hog/predictions.csv');F=readpred('edge','edge_four_cell/results/predictions.csv')
 D=readpred('randumb','randumb/predictions.csv');H=readpred('pa_hog','pa_hog/predictions.csv');P=readpred('pa_full','pa_full/predictions.csv')
 (O/'aggregation_checkpoint.json').write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding='utf8')
 compare('C1/main',base(C,'ideal_plus_multi4')['char'],base(C,'ideal_only')['char'],20260825,'integers',[11.833333333333332,18.555555555555557])
 # Original E1 clusters are sorted by the character string, not numeric class ID.
 with (B/'whole_glyph/E1/predictions.csv').open(encoding='utf-8-sig',newline='') as ef:ecnames={int(x['true_label']):x['character'] for x in csv.DictReader(ef)}
 eorder=sorted(range(180),key=lambda i:ecnames[i])
 compare('E1/main',base(E,'proto_skeleton_ideal_plus_multi4',cond='moderate')['char'][eorder],base(E,'proto_skeleton_ideal_only',cond='moderate')['char'][eorder],20260826,'integers',[12.055555555555555,18.194444444444443])
 # nAUBC uses non-equally spaced budget grid, integrated within character.
 wt=np.array([8,16,24,32,16])/96;budgets=[32,48,64,96,128]
 carr=np.array([[base(Q,m,budget=b)['char'] for b in budgets] for m in ['ideal_only','ideal_plus_multi4']])
 auc=np.einsum('mbc,b->mc',carr,wt);vals=np.column_stack([auc.T,carr[0].T,carr[1].T,(carr[1]-carr[0]).T,auc[1]-auc[0]])
 ci=boot(vals,20260829,'multinomial');ref=readj('whole_glyph/StudyB/main_results.json')
 R['budget']={'accuracy_percent':(carr.mean(2)*100).tolist(),'nAUBC':auc.mean(1).tolist(),'delta_nAUBC':float((auc[1]-auc[0]).mean()),'ci95_delta_nAUBC':ci[:,-1].tolist(),'ci95_nAUBC':ci[:,:2].T.tolist(),'ci95_accuracy_percent':(ci[:,2:12].T*100).tolist(),'ci95_delta_percent':(ci[:,12:17].T*100).tolist()}
 chk('budget/primary_CI',ci[:,-1].tolist(),[ref['bootstrap']['delta_naubc'][k] for k in ['lower','upper']])
 for j,m in enumerate(['ideal_only','ideal_plus_multi4']):
  for k,b in enumerate(budgets):
   rr=next(x for x in ref['bootstrap']['accuracy'] if x['method']==m and x['budget']==b)
   chk(f'budget/{m}/{b}/CI',ci[:,2+5*j+k].tolist(),[rr['lower'],rr['upper']])
 for rr in readj('internal_hog/results/hog/summary.json')['comparisons']:
  compare('internal/'+rr['a']+'-'+rr['b'],base(I,rr['a'])['char'],base(I,rr['b'])['char'],20260909,expected=rr['character_conditional_95ci_pp'])
 edgecontrast={'skeleton_mma_gain':base(F,'skeleton_mma')['char']-base(F,'skeleton_ideal')['char'],'edge_mma_gain':base(F,'edge_mma')['char']-base(F,'edge_ideal')['char'],'skeleton_minus_edge_ideal':base(F,'skeleton_ideal')['char']-base(F,'edge_ideal')['char'],'skeleton_minus_edge_mma':base(F,'skeleton_mma')['char']-base(F,'edge_mma')['char']}
 edgecontrast['interaction']=edgecontrast['skeleton_mma_gain']-edgecontrast['edge_mma_gain']
 for name,d in edgecontrast.items():compare('edge/'+name,d,np.zeros(180),20260909)
 for m in ['skeleton_ideal','skeleton_mma','edge_ideal','edge_mma']:
  compare('edge/cell/'+m,base(F,m)['char'],np.zeros(180),20260909,expected=readj('edge_four_cell/results/summary.json')['cells'][m]['ci95_percent'])
 rd=readj('randumb/results_summary.json');R['randumb_summary']={}
 for panel,pp in rd['panels'].items():
  R['randumb_summary'][panel]={}
  for method,old in pp['methods'].items():
   vg=[v for k,v in R['metrics']['randumb'].items() if k.split('|')[0:2]==[panel,method]];ac=[v['accuracy_percent'] for v in vg]
   st=dict(accuracy_percent=float(np.mean(ac)),seed_sd_percent=float(np.std(ac,ddof=1)) if len(ac)>1 else None,macro_f1=float(np.mean([v['macro_f1_subpanel'] for v in vg])))
   R['randumb_summary'][panel][method]=st
   chk('randumb/'+panel+'/'+method+'/metric',[st['accuracy_percent'],st['macro_f1']],[old['accuracy_percent'],old['macro_f1']])
   if len(ac)>1:chk('randumb/'+panel+'/'+method+'/sd',st['seed_sd_percent'],old['seed_sd_percent'])
  for rr in pp['comparisons']:compare('randumb/'+panel+'/'+rr['a']+'-'+rr['b'],meanchar(D,panel,rr['a']),meanchar(D,panel,rr['b']),20260928,expected=rr['conditional95ci_pp'])
 for tag,gr,seeds in [('pa_hog',H,{'DEV':26092831,'E1':26092832,'C1':26092833}),('pa_full',P,{'E1':26092932,'C1':26092933})]:
  for panel,comparisons in readj(tag+'/paired_comparisons.json').items():
   for rr in comparisons:
    if rr['ci95_pp'] is None:continue
    a,b=rr['comparison'].split('-');compare(tag+'/'+panel+'/'+rr['comparison'],meanchar(gr,panel,a),meanchar(gr,panel,b),seeds[panel],expected=rr['ci95_pp'])
 # Exact cross-experiment frozen HOG replay, all cells (not just aggregate counts).
 for panel,orig,om0,om1,cond in [('C1',C,'ideal_only','ideal_plus_multi4',''),('E1',E,'proto_skeleton_ideal_only','proto_skeleton_ideal_plus_multi4','moderate')]:
  for m,om in [('H0',om0),('H1',om1)]:
   refp=base(orig,om,cond=cond)['p']
   for tag,gg in [('randumb',D),('pa_hog',H),('pa_full',P)]:chk('replay/'+panel+'/'+tag+'/'+m, bool(np.array_equal(base(gg,m,panel=panel)['p'],refp)),True)
 for tag,gg in [('pa_hog',H),('pa_full',P)]:
  for m,om in [('H0','ideal_only'),('H1','mma_g0')]:chk('replay/DEV/'+tag+'/'+m,bool(np.array_equal(base(gg,m,panel='DEV')['p'],base(I,om)['p'])),True)
 # Multi-axis sensitivity independently implements paired product empirical bootstrap.
 # Original seed reused; sampler batching is explicitly independent, so not exact historical RNG replay.
 cubes={'C1':base(C,'ideal_plus_multi4')['cube']-base(C,'ideal_only')['cube'],'E1':base(E,'proto_skeleton_ideal_plus_multi4',cond='moderate')['cube']-base(E,'proto_skeleton_ideal_only',cond='moderate')['cube']}
 cubes['StudyB']=sum(w*(base(Q,'ideal_plus_multi4',budget=b)['cube']-base(Q,'ideal_only',budget=b)['cube']) for w,b in zip(wt,budgets))
 R['multiaxis']={}
 for name,cube in cubes.items():
  rng=np.random.default_rng(20260825 if name=='C1' else 20260829);nf,nm=cube.shape[1:];dist=[]
  for _ in range(50):
   cw=rng.multinomial(180,np.full(180,1/180),size=2000);fw=rng.multinomial(nf,np.full(nf,1/nf),size=2000);mw=rng.multinomial(nm,np.full(nm,1/nm),size=2000)
   dist.append(np.einsum('bc,bf,bm,cfm->b',cw,fw,mw,cube,optimize=True)/(180*nf*nm))
  R['multiaxis'][name]={'point':float(cube.mean()),'ci95':np.quantile(np.concatenate(dist),[.025,.975]).tolist(),'n_fonts':nf,'n_maps':nm,'n_characters':180,'n_replicates':100000,'note':'Independent product empirical bootstrap implementation; not claimed as exact historical RNG replay.'}
 # Exact E1 historical crossed sampler: character-string order, 1000-row batches,
 # sequential integers for characters, fonts, mappings in each batch.
 rng=np.random.default_rng(20260829);cube=cubes['E1'][eorder];dist=[]
 for _ in range(100):
  cc=rng.integers(0,180,size=(1000,180),dtype=np.int64);ff=rng.integers(0,4,size=(1000,4),dtype=np.int64);mm=rng.integers(0,5,size=(1000,5),dtype=np.int64)
  dist.append(cube[cc[:,:,None,None],ff[:,None,:,None],mm[:,None,None,:]].mean((1,2,3)))
 R['multiaxis']['E1_exact_historical_sampler']={'ci95':np.quantile(np.concatenate(dist),[.025,.975]).tolist(),'seed':20260829,'iterations':100000}
 chk('E1/crossed_historical',R['multiaxis']['E1_exact_historical_sampler']['ci95'],[0.09944444444444445,0.20194444444444445])
 # Descriptive negative and heterogeneity results.
 ec=cubes['E1'].mean((1,2));R['heterogeneity']={'E1_characters':[int(np.sum(ec>0)),int(np.sum(ec==0)),int(np.sum(ec<0))]}
 fg=np.array([(base(Q,'ideal_plus_multi4',budget=b)['cube']-base(Q,'ideal_only',budget=b)['cube']).mean((0,2)) for b in budgets]);R['heterogeneity']['budget_font_positive_cells']=int((fg>0).sum());R['heterogeneity']['budget_font_gains_pp']=(fg*100).tolist()
 R['heterogeneity']['96_to128_decreasing_cells']={m:int(np.sum(base(Q,m,budget=128)['cube'].mean(0)<base(Q,m,budget=96)['cube'].mean(0))) for m in ['ideal_only','ideal_plus_multi4']}

R['failed_checks']=[x for x in R['checks'] if not x['passed']]
R['total_prediction_rows']=sum(x['rows'] for x in R['integrity'].values())
record(__file__)
(O/'recomputed.json').write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'rows':R['total_prediction_rows'],'integrity':R['integrity'],'failed_checks':R['failed_checks'],'budget':R['budget'],'multiaxis':R['multiaxis'],'heterogeneity':R['heterogeneity']},ensure_ascii=False,indent=2))
