"""Editable native draw.io + SVG/PDF from a shared scene graph, no generated data.
The PNG render is produced from the PDF using render_figures.py.
draw.io CLI is unavailable on the task host; its native XML is validated separately.
"""
from pathlib import Path
import math, html, base64, io, json, hashlib
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
FONT='/historical/Windows/Fonts/simhei.ttf'
pdfmetrics.registerFont(TTFont('SimHei',FONT))
pdfmetrics.registerFont(TTFont('TimesNewRoman','/historical/Windows/Fonts/times.ttf'))
pdfmetrics.registerFont(TTFont('TimesItalic','/historical/Windows/Fonts/timesi.ttf'))
INK='#253748';TEAL='#007A83';EDGE='#758C9A';PALE='#F2F7FA';WARM='#FFF3E4';ORANGE='#AB5B16'
WIDTH_PT=16.55/2.54*72

class Scene:
    def __init__(self,name,w,h):
        self.name,self.w,self.h=name,w,h;self.k=WIDTH_PT/w
        self.pdf=canvas.Canvas(str(OUT/(name+'.pdf')),pagesize=(WIDTH_PT,h*self.k),initialFontName='SimHei')
        self.pdf.scale(self.k,self.k);self.pdf.setTitle(name)
        self.svg=[f'<svg xmlns="htt/historical/www.w3.org/2000/svg" xmlns:xlink="htt/historical/www.w3.org/1999/xlink" width="16.55cm" height="{h/w*16.55}cm" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>']
        self.xml=ET.Element('mxGraphModel',adaptiveColors='auto',dx=str(w),dy=str(h),grid='0',page='1',pageWidth=str(w),pageHeight=str(h),math='0')
        self.root=ET.SubElement(self.xml,'root');ET.SubElement(self.root,'mxCell',id='0');ET.SubElement(self.root,'mxCell',id='1',parent='0');self.n=1
    def cell(self,value,style,x,y,w,h):
        self.n+=1;c=ET.SubElement(self.root,'mxCell',id=str(self.n),value=value,style=style,vertex='1',parent='1')
        ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'});return c
    def rect(self,x,y,w,h,fill=PALE,stroke=EDGE,lw=1.3):
        self.pdf.setFillColor(HexColor(fill));self.pdf.setStrokeColor(HexColor(stroke));self.pdf.setLineWidth(lw);self.pdf.rect(x,self.h-y-h,w,h,fill=1,stroke=1)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
        self.cell('',f'rounded=0;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};strokeWidth={lw};',x,y,w,h)
    def text(self,x,y,s,size=18,align='center',color=INK,font='SimHei',bold=False):
        # y is a baseline; every backend uses the same font metrics and baseline.
        self.pdf.setFont(font,size);self.pdf.setFillColor(HexColor(color))
        {'left':self.pdf.drawString,'center':self.pdf.drawCentredString,'right':self.pdf.drawRightString}[align](x,self.h-y,s)
        family={'TimesItalic':'Times New Roman','TimesNewRoman':'Times New Roman','SimHei':'SimHei','Arial':'Arial','YaHei':'Microsoft YaHei'}[font]
        italic=' font-style="italic"' if font=='TimesItalic' else ''
        self.svg.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" text-anchor="{dict(left="start",center="middle",right="end")[align]}" fill="{color}"{italic}>{html.escape(s)}</text>')
        width=pdfmetrics.stringWidth(s,font,size)+4
        xx=x if align=='left' else x-width if align=='right' else x-width/2
        self.cell(s,f'text;html=0;whiteSpace=nowrap;overflow=visible;align={align};verticalAlign=middle;fontFamily={family};fontSize={size};fontColor={color};fontStyle={2 if font=="TimesItalic" else 0};spacing=0;strokeColor=none;fillColor=none;',xx,y-size*.96,width,size*1.22)
    def line(self,pts,color=INK,lw=1.4,dash=None,arrow=False):
        self.pdf.setStrokeColor(HexColor(color));self.pdf.setLineWidth(lw);self.pdf.setDash(dash or [])
        p=self.pdf.beginPath();p.moveTo(pts[0][0],self.h-pts[0][1])
        for x,y in pts[1:]:p.lineTo(x,self.h-y)
        self.pdf.drawPath(p);self.pdf.setDash([])
        ds=f' stroke-dasharray="{" ".join(map(str,dash))}"' if dash else ''
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="{lw}"{ds}/>')
        if arrow:
            x,y=pts[-1];px,py=pts[-2];a=math.atan2(y-py,x-px);v=[(x,y),(x-7*math.cos(a-.42),y-7*math.sin(a-.42)),(x-7*math.cos(a+.42),y-7*math.sin(a+.42))]
            p=self.pdf.beginPath();p.moveTo(v[0][0],self.h-v[0][1])
            for xx,yy in v[1:]:p.lineTo(xx,self.h-yy)
            p.close();self.pdf.setFillColor(HexColor(color));self.pdf.drawPath(p,fill=1,stroke=0)
            self.svg.append(f'<polygon points="{" ".join(f"{xx},{yy}" for xx,yy in v)}" fill="{color}"/>')
        self.n+=1;c=ET.SubElement(self.root,'mxCell',id=str(self.n),style=f'edgeStyle=none;html=0;rounded=0;endArrow={"block" if arrow else "none"};endFill=1;endSize=7;strokeColor={color};strokeWidth={lw};dashed={1 if dash else 0};',edge='1',parent='1')
        g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
        ET.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),**{'as':'sourcePoint'})
        ET.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),**{'as':'targetPoint'})
        if len(pts)>2:
            ar=ET.SubElement(g,'Array',**{'as':'points'})
            for x,y in pts[1:-1]:ET.SubElement(ar,'mxPoint',x=str(x),y=str(y))
    def dot(self,x,y,r,color):
        self.pdf.setFillColor(HexColor(color));self.pdf.circle(x,self.h-y,r,fill=1,stroke=0)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')
        self.cell('',f'ellipse;fillColor={color};strokeColor=none;',x-r,y-r,2*r,2*r)
    def image(self,x,y,w,h,array,vmax=1):
        im=Image.fromarray(np.rint(np.clip(array/vmax,0,1)*255).astype('uint8'),'L');buf=io.BytesIO();im.save(buf,format='PNG');data=buf.getvalue();uri='data:image/png;base64,'+base64.b64encode(data).decode()
        self.pdf.drawImage(ImageReader(im),x,self.h-y-h,width=w,height=h)
        self.svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" image-rendering="pixelated" xlink:href="{uri}"/>')
        # draw.io's image style accepts data:image/png,<base64>.
        self.cell('',f'shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image=data:image/png,{base64.b64encode(data).decode()};',x,y,w,h)
    def vector(self,x,y,w,h,color=PALE):
        self.rect(x,y,w,h,color,EDGE,1)
        for j in [1,2,4,5]:self.line([(x,y+h*j/6),(x+w,y+h*j/6)],EDGE,.8)
        for dy in [-4,0,4]:self.dot(x+w/2,y+h*.50+dy,.8,INK)
    def save(self):
        self.pdf.showPage();self.pdf.save()
        (OUT/(self.name+'.svg')).write_text('\n'.join(self.svg+['</svg>']),encoding='utf8')
        ET.indent(self.xml);ET.ElementTree(self.xml).write(OUT/(self.name+'.drawio'),encoding='utf-8',xml_declaration=True)
        root=ET.parse(OUT/(self.name+'.drawio')).getroot();cells=root.findall('./historical/root/mxCell');ids=[c.get('id') for c in cells]
        assert len(ids)==len(set(ids)) and {'0','1'}.issubset(ids)
        assert all(c.find('mxGeometry') is not None for c in cells if c.get('edge')=='1')

class Formula:
    def __init__(self,d):self.d=d;self.items=[];self.x=0
    def t(self,s,size=22,font='TimesNewRoman',dy=0):
        self.items.append((self.x,dy,s,size,font));self.x+=pdfmetrics.stringWidth(s,font,size);return self
    def space(self,n=4):self.x+=n;return self
    def var(self,s,sub=None,sup=None,size=22):
        self.t(s,size,'TimesItalic');start=self.x;end=start
        if sub:self.t(sub,size*.64,'TimesItalic',size*.22);end=max(end,self.x)
        if sup:self.x=start;self.t(sup,size*.64,'TimesItalic',-size*.48);end=max(end,self.x)
        self.x=end;return self
    def mu(self):return self.var('μ',sub='c').t('(').t('R',font='TimesItalic').t(')')
    def frac(self):
        self.items.extend([(self.x+7,-13,'1',15,'TimesNewRoman'),(self.x+2,9,'30',15,'TimesNewRoman'),(self.x,-8,None,25,None)]);self.x+=28;return self
    def sum(self,sub,sup):
        self.items.extend([(self.x+4,0,'∑',33,'TimesNewRoman'),(self.x+11,-25,sup,15,'TimesNewRoman'),(self.x+1,19,sub,15,'TimesItalic')]);self.x+=35;return self
    def render(self,cx,y):
        x=cx-self.x/2
        for dx,dy,s,size,font in self.items:
            if s is None:self.d.line([(x+dx,y+dy),(x+dx+size,y+dy)],INK,1)
            else:self.d.text(x+dx,y+dy,s,size,'left',font=font)

def figure1():
    d=Scene('fig1-method-framework',1020,438)
    d.text(10,26,'(a) 离线原型构建（整字协议）',22,'left')
    boxes=[(10,86),(114,104),(236,340),(594,86),(698,156),(872,138)]
    for i,(x,w) in enumerate(boxes):
        d.rect(x,48,w,180,WARM if i==4 else PALE,ORANGE if i==4 else EDGE)
        if i<5:d.line([(x+w+2,138),(boxes[i+1][0]-3,138)],arrow=True)
    d.text(53,77,'6种',19);d.text(53,101,'训练字体',18)
    a=np.load(ROOT/'figure_data/fig1_simulation_arrays.npz');d.image(24,119,58,58,1-a['glyph'])
    d.text(53,204,'带标签字形',16)
    d.text(166,77,'骨架引导',18);d.text(166,99,'选点',18)
    selected=set(a['selected'].tolist())
    for j,(x,y) in enumerate(a['centers'][0]):
        d.dot(126+x/96*80,111+y/96*80,1.8 if j in selected else 1,ORANGE if j in selected else '#BDC7CF')
    d.text(166,213,'固定预算 B',17)
    d.text(406,77,'固定映射仿真',19)
    vmax=float(a['spv'].max())
    for j in range(5):
        x=247+j*65
        d.text(x+29,102,'理想' if j==0 else f'失配{j}',16)
        d.image(x,111,58,58,a['spv'][j],vmax)
        d.text(x+29,190,f'r = {j}',16,font='TimesNewRoman')
    d.text(406,214,'同一字形复用活跃索引',17)
    d.text(637,77,'HOG特征',18);d.vector(625,103,24,74)
    d.text(637,197,'原始向量',16);d.text(637,217,'4356维',16)
    d.text(776,77,'MMA原型聚合',19,color=ORANGE)
    d.text(776,112,'跨字体、跨映射',17);d.text(776,138,'同类等权均值',18)
    d.line([(776,151),(776,174)],ORANGE,arrow=True)
    d.text(776,201,'原型 L2 归一化',18)
    d.text(941,77,'类别原型库',19)
    for x,label in [(889,'1'),(921,'2'),(976,'180')]:
        d.vector(x,105,18,65)
        f=Formula(d);f.var('P',sub=label,size=21);f.render(x+9,190)
    d.text(956,146,'…',20,font='TimesNewRoman');d.text(941,216,'180类 × 4356维',16)
    d.text(10,277,'(b) 测试识别',22,'left')
    d.line([(941,230),(941,251),(675,251),(675,303)],TEAL,1.7,arrow=True)
    d.text(810,245,'固定原型库',16,color=TEAL)
    lower=[(65,195),(308,195),(551,247),(846,164)]
    for i,(x,w) in enumerate(lower):
        d.rect(x,306,w,105,'#FFFFFF',EDGE)
        if i<3:d.line([(x+w+3,358),(lower[i+1][0]-4,358)],arrow=True)
    d.text(162,335,'测试 SPV 图像',20);d.text(162,365,'新字体 · 未见映射',18);d.text(162,392,'分类器仅接收图像',16)
    d.text(405,335,'一次 HOG 提取',20);d.text(405,366,'查询向量 L2 归一化',18)
    f=Formula(d);f.var('q',size=21).t(' = ',21).var('h',size=21).t(' / ‖h‖',21).t('2',13.44,dy=4.62);f.render(405,394)
    d.text(675,339,'与各类别原型计算',20);d.text(675,370,'余弦相似度',20)
    f=Formula(d);f.var('P',sub='c',sup='T',size=21).space().var('q',size=21);f.render(675,396)
    d.text(928,342,'取最大分数',20);d.text(928,374,'输出预测类别',20)
    d.save()

def figure2():
    d=Scene('fig2-prototype-aggregation',1020,377)
    d.text(225,28,'(a) 同一类别 c 的特征观测',22)
    d.text(610,28,'(b) 聚合与归一化',22)
    d.text(902,28,'(c) 固定规模原型库',22)
    for j in range(5):
        x=86+j*65;d.text(x+28,61,'理想' if j==0 else f'失配{j}',17);d.text(x+28,81,f'r = {j}',17,font='TimesNewRoman')
    for i in range(6):
        y=95+i*34;d.text(70,y+20,f'字体{i+1}',17,'right')
        for j in range(5):
            x=86+j*65;d.rect(x,y,56,28,'#E2EEF5' if j==0 else '#FCEBD5',EDGE,1)
            f=Formula(d);f.var('h',sup=f'c,{i+1},{j}',size=21);f.render(x+28,y+20)
    d.text(245,321,'6字体 × 5映射 = 30个向量',18)
    d.text(245,348,'每个向量均为 4356 维',18)
    d.line([(414,196),(441,196),(441,120),(457,120)],arrow=True)
    d.rect(462,88,296,61,WARM,ORANGE)
    d.text(610,113,'同类原始 HOG 向量',20);d.text(610,139,'等权求均值',20)
    f=Formula(d);f.mu().space().t('=').space().frac().sum('f=1','6').sum('r=0','4').space().var('h',sup='c,f,r');f.render(610,192)
    d.line([(610,216),(610,237)],arrow=True)
    d.rect(462,242,296,55,WARM,ORANGE);d.text(610,266,'均值向量',20);d.text(610,290,'L2 归一化',20)
    f=Formula(d);f.var('P',sub='c').t('(').t('R',font='TimesItalic').t(')').space().t('=').space().mu().t('/‖');f.mu().t('‖₂');f.render(610,343)
    d.line([(764,269),(795,269)],arrow=True)
    d.text(908,78,'每行一个类别原型',18)
    for y,label in [(115,'1'),(156,'2'),(238,'180')]:
        f=Formula(d);f.var('P',sub=label,size=21);f.render(819,y+19)
        for j in range(5):d.rect(842+j*30,y,30,27,PALE,EDGE,1)
        d.text(917,y+20,'…',18,font='TimesNewRoman')
    for y in [201,208,215]:d.dot(906,y,1.1,INK)
    d.text(919,290,'4356维',18)
    d.text(908,324,'180类 × 4356维',20)
    d.text(908,350,'两种方法的原型规模相同',16)
    d.save()

if __name__=='__main__':figure1();figure2()
