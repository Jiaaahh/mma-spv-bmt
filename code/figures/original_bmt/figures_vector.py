"""Render frozen scientific displays through the retained vector scene graph."""
from pathlib import Path
import shutil,importlib.util,json,re,csv
import numpy as np
from PIL import Image
import pypdfium2 as pdfium
W=Path(__file__).resolve().parent
SRC=Path('/historical/user/Documents/Codex/2026-11-05/thread-01a1a76f-34b6-72a1-8dcc-6498b042475f/outputs/论文配图与源文件_20261105')
OUT=W.parent/'outputs'/'BMT双语论文_20261107'/'figures';OUT.mkdir(parents=True,exist_ok=True)
DATA=W/'figure_data';shutil.copytree(SRC/'data',DATA,dirs_exist_ok=True)
code=(W/'prepare_figures.py').read_text('utf8');trans=code[code.index('TRAN={'):code.index('\nsrc=')];exec(trans)
src=(SRC/'source/draw_methods.py').read_text('utf8').replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path(__file__).resolve().parent').replace("ROOT/'data/fig1_simulation_arrays.npz'","ROOT/'figure_data/fig1_simulation_arrays.npz'")
src=src.replace("'SimHei':'SimHei'", "'SimHei':'SimHei','Arial':'Arial','YaHei':'Microsoft YaHei'")
(W/'method_scene.py').write_text(src,encoding='utf8')
spec=importlib.util.spec_from_file_location('scene',W/'method_scene.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.pdfmetrics.registerFont(m.TTFont('Arial','/historical/Windows/Fonts/arial.ttf'))
m.pdfmetrics.registerFont(m.TTFont('YaHei','/historical/Windows/Fonts/msyh.ttc'))
TRAN.update({'6种':'6','训练字体':'fonts','带标签字形':'Glyphs','类别原型库':'Prototypes','一次 HOG 提取':'Extract HOG','查询向量 L2 归一化':'Normalize query','(c) 固定规模原型库':'(C) Library','每行一个类别原型':'One class per row','两种方法的原型规模相同':'Same library size'})
orig=m.Scene.text;BASE='#526574';MMA='#007A83';DELTA='#825515'
def tx(d,x,y,t,size=17,align='center',color=m.INK):d.text(x,y,t,size,align,color)
def axis(d,box,xlim,ylim,xticks,yticks,xlabels=None):
 l,t,w,h=box;xf=lambda x:l+(x-xlim[0])/(xlim[1]-xlim[0])*w;yf=lambda y:t+h-(y-ylim[0])/(ylim[1]-ylim[0])*h
 for y in yticks:d.line([(l,yf(y)),(l+w,yf(y))],'#DFE5E8',.9);tx(d,l-10,yf(y)+5,str(y),16,'right')
 d.line([(l,t),(l,t+h),(l+w,t+h)],m.INK,1.2)
 for i,x in enumerate(xticks):
  d.line([(xf(x),t+h),(xf(x),t+h+5)],m.INK,1)
  lab=xlabels[i] if xlabels else str(x)
  for j,s in enumerate(lab.split('\n')):tx(d,xf(x),t+h+25+j*18,s,16)
 return xf,yf
def marker(d,x,y,color,square=False):
 if square:d.rect(x-4,y-4,8,8,color,color,.5)
 else:d.dot(x,y,4.5,color)
def err(d,x,y,lo,hi,xf,yf,color,square=False):
 xx=xf(x);d.line([(xx,yf(lo)),(xx,yf(hi))],color,1.3)
 for v in [lo,hi]:d.line([(xx-4,yf(v)),(xx+4,yf(v))],color,1.3)
 marker(d,xx,yf(y),color,square)
def read(name):
 return [{k:float(v) if k not in ['condition','condition_id'] else v for k,v in r.items()} for r in csv.DictReader((DATA/name).open(encoding='utf-8-sig'))]
def result2(lang):
 d=m.Scene('Figure2',1020,425);en=lang=='en';r=read('budget_source_data.csv');xs=[x['budget'] for x in r]
 tx(d,510,24,'n = 3600 per method/budget; character-cluster conditional 95% intervals' if en else '每方法每预算 n = 3600；误差线为字符簇条件95%区间',17)
 tx(d,270,62,'(A) Accuracy by budget' if en else '(A) 准确率—预算曲线',21)
 tx(d,788,62,'(B) Paired accuracy gain' if en else '(B) 配对准确率增益',21)
 marker(d,120,90,BASE,True);tx(d,134,96,'Ideal prototype' if en else '理想原型',16,'left',BASE)
 marker(d,325,90,MMA);tx(d,339,96,'MMA',16,'left',MMA)
 xf,yf=axis(d,(80,126,380,208),(27,133),(30,65),xs,[30,40,50,60],list(map(lambda x:str(int(x)),xs)))
 tx(d,80,118,'Accuracy, %' if en else '准确率，%',17,'left')
 for p,c,sq,dash in [('baseline',BASE,True,[5,3]),('mma',MMA,False,None)]:
  d.line([(xf(z['budget']),yf(z[p])) for z in r],c,1.8,dash)
  for z in r:err(d,z['budget'],z[p],z[p+'_lo'],z[p+'_hi'],xf,yf,c,sq)
 xg,yg=axis(d,(608,126,365,208),(27,133),(-1,23),xs,[0,5,10,15,20],list(map(lambda x:str(int(x)),xs)))
 d.line([(608,yg(0)),(973,yg(0))],m.INK,1.3);tx(d,608,118,'MMA − ideal, pp' if en else 'MMA − 理想原型，百分点',17,'left')
 for z in r:err(d,z['budget'],z['delta'],z['delta_lo'],z['delta_hi'],xg,yg,DELTA);tx(d,xg(z['budget']),yg(z['delta_hi'])-10,f"+{z['delta']:.2f}",16,color=DELTA)
 for x in [270,790]:tx(d,x,388,'Activation budget B' if en else '活跃位置预算 B',18)
 tx(d,510,420,('Primary ' if en else '主终点 ')+ 'ΔnAUBC = 0.1010 [0.0818, 0.1206]',18)
 d.save()
def result3(lang):
 d=m.Scene('Figure3',1020,430);en=lang=='en';r=read('mismatch_source_data.csv');xs=list(range(5));labels=['Mild','Moderate','Severe','Position\nonly','Scale\nonly'] if en else ['轻度','中度','重度','仅位置','仅尺度']
 tx(d,510,24,'n = 3600 per condition; fixed training prototypes; point estimates only' if en else '每条件 n = 3600；训练原型固定；仅示点估计',17)
 tx(d,270,62,'(A) Accuracy' if en else '(A) 准确率',21);tx(d,790,62,'(B) Paired gain' if en else '(B) 配对增益',21)
 marker(d,120,90,BASE,True);tx(d,134,96,'Ideal prototype' if en else '理想原型',16,'left',BASE);marker(d,325,90,MMA);tx(d,339,96,'MMA',16,'left',MMA)
 xf,yf=axis(d,(75,126,405,205),(-.5,4.5),(25,105),xs,[25,50,75,100],labels);tx(d,75,118,'Accuracy, %' if en else '准确率，%',17,'left')
 for i,z in enumerate(r):
  d.line([(xf(i-.10),yf(z['baseline'])),(xf(i+.10),yf(z['mma']))],'#ABB7BD',1.1)
  base_above=z['baseline']>z['mma']
  for p,col,off,sq,dy in [('baseline',BASE,-.10,True,-12 if base_above else 21),('mma',MMA,.10,False,21 if base_above else -12)]:
   marker(d,xf(i+off),yf(z[p]),col,sq);tx(d,xf(i+off),yf(z[p])+dy,f'{z[p]:.2f}',16,color=col)
 xg,yg=axis(d,(620,126,355,205),(-.5,4.5),(-10,38),xs,[-10,0,10,20,30],labels);tx(d,620,118,'MMA − ideal, pp' if en else 'MMA − 理想原型，百分点',17,'left');d.line([(620,yg(0)),(975,yg(0))],m.INK,1.3)
 for i,z in enumerate(r):
  y=z['delta'];color=DELTA if y>=0 else BASE;d.rect(xg(i)-18,min(yg(y),yg(0)),36,abs(yg(y)-yg(0)),color,color,.3);tx(d,xg(i),yg(y)+(-10 if y>=0 else 22),f'{y:+.2f}',17,color=color)
 for x in [270,790]:tx(d,x,410,'Test mismatch condition' if en else '测试失配条件',18)
 d.save()
def cases(lang):
 en=lang=='en';d=m.Scene('Figure4',1020,500);cs=json.loads((DATA/'fig5_cases/provenance.json').read_text('utf8'))['cases'];a=[np.load(DATA/f'fig5_cases/case_{i+1}.npz') for i in range(4)];vmax=max(float(z['spv'].max()) for z in a)
 tx(d,320,24,'(A) Corrected by MMA' if en else '(A) 理想原型误识别，MMA改正',21);tx(d,780,24,'(B) Regressed with MMA' if en else '(B) 理想原型正确，MMA误识别',21)
 tx(d,49,147,'Window' if en else '观看窗口',17);tx(d,49,171,'80 × 80',16);tx(d,49,320,'SPV',18);tx(d,49,344,'96 × 96',16)
 # Character labels use Chinese font while scientific samples remain unchanged.
 for i,(c,z) in enumerate(zip(cs,a)):
  x=128+i*222;cx=x+80
  tx(d,cx,62,('True: ' if en else '真值：')+c['character'],19)
  d.image(x,80,160,160,z['crop'],1);d.image(x,258,160,160,z['spv'],vmax)
  tx(d,cx,445,('Ideal: ' if en else '理想：')+c['ideal_prediction_character'],18)
  tx(d,cx,470,'MMA: '+c['mma_prediction_character'],18);tx(d,cx,493,'ST Hupo' if en else '华文琥珀',16)
 d.save()
for lang in ['en','zh']:
 m.OUT=OUT/lang;m.OUT.mkdir(exist_ok=True)
 def translated(self,x,y,s,size=18,align='center',color=m.INK,font='SimHei',bold=False):
  if lang=='en':
   s=TRAN.get(s,s);s=re.sub(r'失配(\d)',r'Map \1',s);s=re.sub(r'字体(\d)',r'Font \1',s)
   if font=='SimHei':font='YaHei' if re.search('[\u4e00-\u9fff]',s) else 'Arial'
  else:s=s.replace('(a)','(A)').replace('(b)','(B)').replace('(c)','(C)')
  s=s.replace(' − ',' - ')
  return orig(self,x,y,s,size,align,color,font,bold)
 m.Scene.text=translated;m.figure1();m.figure2()
 for old,new in [('fig1-method-framework','Figure1'),('fig2-prototype-aggregation','FigureS1')]:
  for ext in ['pdf','svg','drawio']:shutil.move(m.OUT/f'{old}.{ext}',m.OUT/f'{new}.{ext}')
 result2(lang);result3(lang);cases(lang)
 for p in m.OUT.glob('*.pdf'):
  pdf=pdfium.PdfDocument(str(p));page=pdf[0];im=page.render(scale=600/72).to_pil().convert('RGB');im.save(p.with_suffix('.png'),dpi=(600,600));page.close();pdf.close()
  im.thumbnail((1600,1600));pv=W/'figure_previews'/lang;pv.mkdir(parents=True,exist_ok=True);im.save(pv/p.with_suffix('.png').name)
 for p in m.OUT.glob('*.svg'):
  s=p.read_text('utf8').replace('SimHei','Microsoft YaHei' if p.stem=='Figure4' or lang=='zh' else 'Arial');p.write_text(s,encoding='utf8')
 print(lang,'figures complete')
