from pathlib import Path
import urllib.request,hashlib,json,time,zipfile
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];DATA=ROOT/'work/caltech101_official'
DATA.mkdir(exist_ok=True);target=DATA/'caltech-101.zip'
url='http/historical/data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1'
if not target.exists():
    req=urllib.request.Request(url+'&nocache='+str(int(time.time())),headers={'User-Agent':'MMA-research-sanity-check','Cache-Control':'no-cache'})
    with urllib.request.urlopen(req,timeout=60) as r,target.with_suffix('.partial').open('wb') as f:
        size=0
        while True:
            b=r.read(4*1024**2)
            if not b:break
            f.write(b);size+=len(b);print(f'DOWNLOAD {size//1024**2} MiB',flush=True)
    target.with_suffix('.partial').replace(target)
data=target.read_bytes();assert hashlib.md5(data).hexdigest()=='3138e1922a9193bfa496528edbbc45d0'
with zipfile.ZipFile(target) as z:names=z.namelist()
(OUT/'dataset_source.json').write_text(json.dumps(dict(url=url,md5=hashlib.md5(data).hexdigest(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),members=names),indent=2),encoding='utf8')
print('ARCHIVE VERIFIED',names,flush=True)
