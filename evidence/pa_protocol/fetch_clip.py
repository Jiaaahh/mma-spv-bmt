from pathlib import Path
import urllib.request, json, hashlib, zipfile, io, ast
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'MMA-research-preflight'})
    return urllib.request.urlopen(req,timeout=60)
commit=json.load(get('http/historical/api.github.com/repos/openai/CLIP/commits/main'))['sha']
url=f'http/historical/codeload.github.com/openai/CLIP/zip/{commit}'
data=get(url).read()
dest=ROOT/'work/proto_adapter_clip_source'; dest.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for n in z.namelist():
        p=Path(*Path(n).parts[1:]); target=(dest/p).resolve()
        assert target.is_relative_to(dest.resolve())
        if n.endswith('/'): target.mkdir(parents=True,exist_ok=True)
        elif p.parts:
            target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(n))
tree=ast.parse((dest/'clip/clip.py').read_text(encoding='utf-8'))
models=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MODELS' for t in n.targets))
url=models['RN50']; expected=url.split('/')[-2]
weights=ROOT/'work/proto_adapter_clip_weights'; weights.mkdir(exist_ok=True)
target=weights/'RN50.pt'
if not target.exists():
    tmp=weights/'RN50.pt.partial'
    with get(url) as response,tmp.open('wb') as f:
        done=0
        while True:
            b=response.read(4*1024**2)
            if not b: break
            f.write(b); done+=len(b); print(f'DOWNLOAD {done//1024**2} MiB',flush=True)
    digest=hashlib.sha256(tmp.read_bytes()).hexdigest(); assert digest==expected
    tmp.replace(target)
digest=hashlib.sha256(target.read_bytes()).hexdigest(); assert digest==expected
manifest=dict(source='http/historical/github.com/openai/CLIP',commit=commit,archive_sha256=hashlib.sha256(data).hexdigest(),
    weight_url=url,weight_sha256=digest,weight_bytes=target.stat().st_size,weight_verified=True,
    source_files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()})
(OUT/'clip_source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('SOURCE AND WEIGHT VERIFIED',flush=True)
