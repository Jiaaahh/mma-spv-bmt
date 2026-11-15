from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent/'python_deps'))
import shutil,importlib.util,json,re,os
import numpy as np
from PIL import Image
import pypdfium2 as pdfium
W=Path(__file__).resolve().parent
SRC=Path('/historical/user/Documents/Codex/2026-11-05/thread-01a1a76f-34b6-72a1-8dcc-6498b042475f/outputs/论文配图与源文件_20261105')
OUT=W.parent/'outputs'/'BMT双语论文_20261107'/'figures'
OUT.mkdir(parents=True,exist_ok=True)
DATA=W/'figure_data';shutil.copytree(SRC/'data',DATA,dirs_exist_ok=True)
os.environ['MPLCONFIGDIR']=str(W/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
CN=FontProperties(fname='/historical/Windows/Fonts/msyh.ttc').get_name()
STYLE={'font.family':['Arial',CN],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'}
BASE='#526574';MMA='#007A83';DELTA='#825515'
TRAN={
'(a) 离线原型构建（整字协议）':'(A) Offline prototype construction (whole-glyph protocol)',
'6种':'6 training','训练字体':'fonts','带标签字形':'Labeled glyphs','骨架引导':'Skeleton','选点':'selection','固定预算 B':'Budget B',
'固定映射仿真':'Fixed mapping simulation','理想':'Ideal','同一字形复用活跃索引':'Shared activation indices per glyph',
'HOG特征':'HOG','原始向量':'Raw vector','4356维':'4356 dims',
'MMA原型聚合':'MMA aggregation','跨字体、跨映射':'Across fonts/maps','同类等权均值':'Class mean','原型 L2 归一化':'Prototype L2',
'类别原型库':'Class prototypes','180类 × 4356维':'180 × 4356',
'(b) 测试识别':'(B) Test recognition','固定原型库':'Frozen prototype library',
'测试 SPV 图像':'Test SPV image','新字体 · 未见映射':'New font / unseen map','分类器仅接收图像':'Image-only classifier',
'一次 HOG 提取':'One HOG extraction','查询向量 L2 归一化':'Query L2 normalization',
'与各类别原型计算':'Compare with prototypes','余弦相似度':'Cosine similarity','取最大分数':'Maximum score','输出预测类别':'Predicted class',
'(a) 同一类别 c 的特征观测':'(A) Features for one class c','(b) 聚合与归一化':'(B) Aggregation','(c) 固定规模原型库':'(C) Prototype library',
'6字体 × 5映射 = 30个向量':'6 fonts × 5 maps = 30 vectors','每个向量均为 4356 维':'4356 dimensions per vector',
'同类原始 HOG 向量':'Raw HOG from one class','等权求均值':'Equally weighted mean','均值向量':'Class mean','L2 归一化':'L2 normalization',
'每行一个类别原型':'One class prototype per row','两种方法的原型规模相同':'Same size for both methods',
}
src=(SRC/'source'/'draw_methods.py').read_text('utf8')
src=src.replace("ROOT=Path(__file__).resolve().parents[1]", "ROOT=Path(__file__).resolve().parent")
src=src.replace("ROOT/'data/fig1_simulation_arrays.npz'","ROOT/'figure_data/fig1_simulation_arrays.npz'")
local=W/'method_scene.py';local.write_text(src,encoding='utf8')
spec=importlib.util.spec_from_file_location('method_scene',local);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.pdfmetrics.registerFont(mod.TTFont('Arial','/historical/Windows/Fonts/arial.ttf'))
orig_text=mod.Scene.text
def methods(lang):
 mod.OUT=OUT/lang;mod.OUT.mkdir(exist_ok=True)
 def text(self,x,y,s,size=18,align='center',color=mod.INK,font='SimHei',bold=False):
  if lang=='en':
   s=TRAN.get(s,s)
   s=re.sub(r'失配(\d)',r'Map \1',s);s=re.sub(r'字体(\d)',r'Font \1',s)
   # Keep backend font maps intact by registering Arial under the scene's text alias.
  else:s=s.replace('(a)','(A)').replace('(b)','(B)').replace('(c)','(C)')
  return orig_text(self,x,y,s,size,align,color,font,bold)
 if lang=='en':mod.pdfmetrics.registerFont(mod.TTFont('SimHei','/historical/Windows/Fonts/arial.ttf'))
 else:mod.pdfmetrics.registerFont(mod.TTFont('SimHei','/historical/Windows/Fonts/simhei.ttf'))
 mod.Scene.text=text
 mod.figure1();mod.figure2()
 for old,new in [('fig1-method-framework','Figure1'),('fig2-prototype-aggregation','FigureS1')]:
  for ext in ['pdf','svg','drawio']:
   target=mod.OUT/f'{new}.{ext}';shutil.move(mod.OUT/f'{old}.{ext}',target)
   if lang=='en' and ext in ['svg','drawio']:
    target.write_text(target.read_text('utf8').replace('SimHei','Arial'),encoding='utf8')
  pdf=pdfium.PdfDocument(str(mod.OUT/f'{new}.pdf'));page=pdf[0];im=page.render(scale=600/72).to_pil().convert('RGB');im.save(mod.OUT/f'{new}.png',dpi=(600,600));page.close();pdf.close()
def save(fig,dest):
 for ext in ['png','svg','pdf']:fig.savefig(dest.with_suffix('.'+ext),dpi=600,bbox_inches=None)
 plt.close(fig)
def plots(lang):
 import csv
 rows=list(csv.DictReader((DATA/'budget_source_data.csv').open(encoding='utf-8-sig')))
 r=[{k:float(v) for k,v in row.items()} for row in rows];x=np.array([z['budget'] for z in r])
 with plt.rc_context(STYLE):
  fig,axs=plt.subplots(1,2,figsize=(6.52,2.6));fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.75,wspace=.42)
  a,b=axs
  for p,col,marker,ls,label in [('baseline',BASE,'s','--','Ideal prototype' if lang=='en' else '理想原型'),('mma',MMA,'o','-','MMA')]:
   y=np.array([z[p] for z in r]);lo=np.array([z[p+'_lo'] for z in r]);hi=np.array([z[p+'_hi'] for z in r])
   a.errorbar(x,y,yerr=[y-lo,hi-y],fmt=marker,linestyle=ls,markersize=3.7,color=col,lw=1,capsize=2,label=label,elinewidth=.7)
  a.set(title='(A) Accuracy by budget' if lang=='en' else '(A) 准确率—预算曲线',ylabel='Accuracy, %' if lang=='en' else '准确率，%',ylim=(30,65),yticks=[30,40,50,60],xlim=(27,133))
  y=np.array([z['delta'] for z in r]);lo=np.array([z['delta_lo'] for z in r]);hi=np.array([z['delta_hi'] for z in r])
  b.errorbar(x,y,yerr=[y-lo,hi-y],fmt='D',markersize=3.5,color=DELTA,capsize=2,lw=.8)
  for xx,yy,hh in zip(x,y,hi):b.annotate(f'+{yy:.2f}',(xx,hh),xytext=(0,4),textcoords='offset points',ha='center',fontsize=7.2,color=DELTA)
  b.axhline(0,color='#444444',lw=.7);b.set(title='(B) Paired accuracy gain' if lang=='en' else '(B) 配对准确率增益',ylabel='MMA − ideal, pp' if lang=='en' else 'MMA − 理想原型，百分点',ylim=(-1,23),yticks=[0,5,10,15,20],xlim=(27,133))
  for ax in axs:ax.set_xticks(x);ax.set_xlabel('Activation budget B' if lang=='en' else '活跃位置预算 B');ax.grid(axis='y',color='#E0E5E8',lw=.5);ax.set_axisbelow(True)
  fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.27,.92),frameon=False,ncol=2,fontsize=7.5)
  fig.text(.5,.97,'n = 3600 per method/budget; character-cluster conditional 95% intervals' if lang=='en' else '每方法每预算 n = 3600；字符簇条件95%区间',ha='center',fontsize=7.5)
  fig.text(.5,.03,'Primary ΔnAUBC = 0.1010 [0.0818, 0.1206]' if lang=='en' else '主终点 ΔnAUBC = 0.1010 [0.0818, 0.1206]',ha='center',fontsize=8)
  save(fig,OUT/lang/'Figure2')
 rows=list(csv.DictReader((DATA/'mismatch_source_data.csv').open(encoding='utf-8-sig')))
 r=[{k:float(v) if k not in ['condition','condition_id'] else v for k,v in row.items()} for row in rows];x=np.arange(5)
 labels=['Mild','Moderate','Severe','Position\nonly','Scale\nonly'] if lang=='en' else ['轻度','中度','重度','仅位置','仅尺度']
 with plt.rc_context(STYLE):
  fig,axs=plt.subplots(1,2,figsize=(6.52,2.7));fig.subplots_adjust(left=.085,right=.98,bottom=.26,top=.77,wspace=.38)
  a,b=axs
  for p,col,marker,dx,lab in [('baseline',BASE,'s',-.11,'Ideal prototype' if lang=='en' else '理想原型'),('mma',MMA,'o',.11,'MMA')]:
   y=[z[p] for z in r];a.scatter(x+dx,y,s=17,marker=marker,c=col,label=lab,zorder=3)
   for xx,yy in zip(x+dx,y):a.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,7 if p=='baseline' else -11),textcoords='offset points',ha='center',fontsize=6.8,color=col)
  for i,z in enumerate(r):a.plot([i-.11,i+.11],[z['baseline'],z['mma']],color='#B8BFC3',lw=.6,zorder=1)
  a.set(title='(A) Accuracy' if lang=='en' else '(A) 准确率',ylim=(25,103),yticks=[25,50,75,100],ylabel='Accuracy, %' if lang=='en' else '准确率，%')
  deltas=[z['delta'] for z in r];b.bar(x,deltas,width=.55,color=[DELTA if z>=0 else BASE for z in deltas]);b.axhline(0,color='#444',lw=.7)
  for xx,yy in zip(x,deltas):b.text(xx,yy+(1.4 if yy>=0 else -1.5),f'{yy:+.2f}',ha='center',va='bottom' if yy>=0 else 'top',fontsize=7.3)
  b.set(title='(B) Paired gain' if lang=='en' else '(B) 配对增益',ylim=(-10,38),yticks=[-10,0,10,20,30],ylabel='MMA − ideal, pp' if lang=='en' else 'MMA − 理想原型，百分点')
  for ax in axs:ax.set_xticks(x,labels);ax.set_xlim(-.5,4.5);ax.set_xlabel('Test mismatch condition' if lang=='en' else '测试失配条件');ax.grid(axis='y',color='#E0E5E8',lw=.5);ax.set_axisbelow(True)
  fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.29,.91),ncol=2,frameon=False,fontsize=7.5)
  fig.text(.5,.97,'n = 3600 per condition; fixed prototypes; point estimates only' if lang=='en' else '每条件 n = 3600；训练原型固定；仅示点估计',ha='center',fontsize=7.5)
  save(fig,OUT/lang/'Figure3')
def cases(lang):
 provenance=json.loads((DATA/'fig5_cases/provenance.json').read_text('utf8'));cs=provenance['cases'];arrays=[np.load(DATA/f'fig5_cases/case_{i+1}.npz') for i in range(4)];vmax=max(float(a['spv'].max()) for a in arrays)
 with plt.rc_context(STYLE):
  fig=plt.figure(figsize=(6.52,3.42));fig.text(.28,.97,'(A) Corrected by MMA' if lang=='en' else '(A) 理想原型误识别，MMA改正',ha='center',fontsize=9);fig.text(.76,.97,'(B) Regressed with MMA' if lang=='en' else '(B) 理想原型正确，MMA误识别',ha='center',fontsize=9)
  fig.text(.015,.70,'Window\n80 × 80' if lang=='en' else '观看窗口\n80 × 80',fontsize=7.2,va='center');fig.text(.015,.37,'SPV\n96 × 96',fontsize=7.2,va='center')
  for i,(c,a) in enumerate(zip(cs,arrays)):
   left=.13+i*.218
   fig.text(left+.085,.89,('True: ' if lang=='en' else '真值：')+c['character'],fontfamily=CN,ha='center',fontsize=9)
   for name,bottom,mx in [('crop',.56,1),('spv',.24,vmax)]:
    ax=fig.add_axes([left,bottom,.17,.31]);ax.imshow(a[name],cmap='gray',vmin=0,vmax=mx,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   fig.text(left+.085,.18,('Ideal: ' if lang=='en' else '理想：')+c['ideal_prediction_character'],fontfamily=CN,ha='center',fontsize=8)
   fig.text(left+.085,.12,'MMA: '+c['mma_prediction_character'],fontfamily=CN,ha='center',fontsize=8)
   fig.text(left+.085,.055,'ST Hupo' if lang=='en' else '华文琥珀',ha='center',fontsize=7.5)
  save(fig,OUT/lang/'Figure4')
for lang in ['en','zh']:
 methods(lang);plots(lang);cases(lang)
 for p in (OUT/lang).glob('*.png'):
  im=Image.open(p);im.thumbnail((1600,1600));dest=W/'figure_previews'/lang;dest.mkdir(parents=True,exist_ok=True);im.save(dest/p.name)
print('Figures exported',OUT)
