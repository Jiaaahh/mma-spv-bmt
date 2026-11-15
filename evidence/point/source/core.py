"""Point-SPV fixed-budget development adaptation. See protocol and licenses."""
from pathlib import Path
import hashlib, json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from budget_adapter import fixed_budget, render
from vendor.point_models import Representer
from vendor.resnet_arch import ResNet, Bottleneck

ROOT=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def state_hash(module):
    h=hashlib.sha256()
    for name,t in module.state_dict().items():
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def verify_manifest():
    for name,digest in json.loads((ROOT/'manifest.json').read_text(encoding='utf8')).items():
        require(sha(ROOT/name)==digest,'Package changed: '+name)

class GlyphData:
    def __init__(self, role='train'):
        require(role in ['train','test'],'data role')
        self.p=json.loads((ROOT/'protocol.json').read_text())
        self.x=np.load(ROOT/f'data/{role}/glyphs.npy');self.m=np.load(ROOT/f'data/{role}/metadata.npz')
        self.y=self.m['labels'];self.fonts=self.m['fonts']
        require(self.x.shape==(180*len(self.p[role+'_fonts']),96,96) and self.x.dtype==np.float32,'glyph contract')
        require(np.isfinite(self.x).all() and self.x.min()>=0 and self.x.max()<=1,'glyph range')
        require(set(self.fonts)==set(self.p[role+'_fonts']),'font isolation')
        for f in set(self.fonts):require(np.array_equal(np.sort(self.y[self.fonts==f]),np.arange(180)),'classes')
        self.train=np.arange(len(self.x)) if role=='train' else np.empty(0,dtype=np.int64)
    def point(self,i,rng):
        pts=np.argwhere(self.x[i]>self.p['foreground_threshold'])
        require(len(pts)>0,'empty foreground')
        r=self.p['crop_size']//2
        interior=pts[((pts>=r)&(pts<96-r)).all(1)]
        if len(interior):pts=interior
        return pts[int(rng.integers(len(pts)))]
    def patch(self,i,point):
        size=self.p['crop_size'];r=size//2;y,x=map(int,point)
        require(0<=y<96 and 0<=x<96,'point bounds')
        padded=np.pad(self.x[i],r)
        crop=padded[y:y+size,x:x+size].copy()
        yy,xx=np.mgrid[:size,:size];circle=(yy-r)**2+(xx-r)**2<=r*r
        crop*=circle
        retained=float(crop.sum(dtype=np.float64)/self.x[i].sum(dtype=np.float64))
        t=torch.from_numpy(crop)[None,None]
        t=F.interpolate(t,size=(128,128),mode='bilinear',align_corners=False)[0].repeat(3,1,1)
        return t,retained
    def batch(self,rows,rng=None,points=None):
        if points is None:points=np.stack([self.point(int(i),rng) for i in rows])
        pairs=[self.patch(int(i),pt) for i,pt in zip(rows,points)]
        return torch.stack([x[0] for x in pairs]),torch.as_tensor(self.y[rows]),points,np.array([x[1] for x in pairs])

class Encoder(Representer):
    def __init__(self):
        super().__init__(arryaOut=False)
        # Original array-output branch is unused; remove its unused parameters.
        del self.lin1,self.lin2
    def forward(self,x):
        for name in ['seq1','seq2','seq3','seq4','resBlk1','resBlk2','resBlk3','resBlk4','seq5','seq6','seq7','conv8','conv9']:
            x=getattr(self,name)(x)
        return fixed_budget(x.flatten(1),64)

class System(nn.Module):
    def __init__(self,weights=None):
        super().__init__()
        blind=ResNet(Bottleneck,[3,8,36,3]);sighted=ResNet(Bottleneck,[3,8,36,3])
        if weights:
            require(sha(weights).startswith('394f9c45'),'Unexpected pretrained weights (expected ImageNet V1)')
            state=torch.load(weights,map_location='cpu',weights_only=True)
            blind.load_state_dict(state,strict=True);sighted.load_state_dict(state,strict=True)
        self.encoder=Encoder()
        self.teacher=nn.Sequential(*list(sighted.children())[:6]).requires_grad_(False).eval()
        self.channel=nn.Conv2d(1,3,3,padding=1)
        self.blind_feat=nn.Sequential(*list(blind.children())[:6])
        self.classifier=nn.Sequential(*list(blind.children())[6:-1],nn.Flatten(),blind.fc,nn.Linear(1000,180))
    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self
    def forward(self,x,kernels):
        stimulation=self.encoder(x)
        require(torch.equal(stimulation.sum(1),torch.full((len(x),),64.,device=x.device)),'budget violation')
        spv=render(stimulation,kernels)
        vision=F.interpolate(spv,size=(128,128),mode='bilinear',align_corners=False)
        features=self.blind_feat(self.channel(vision))
        with torch.no_grad():target=self.teacher(x)
        require(features.shape==target.shape,'perceptual shape mismatch')
        return self.classifier(features),F.mse_loss(features,target),stimulation,spv

def loss_parts(output,y):
    logits,perceptual,stim,spv=output
    ce=F.cross_entropy(logits,y)
    return .75*perceptual+.25*ce,ce,perceptual
