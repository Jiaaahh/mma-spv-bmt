from pathlib import Path
import hashlib,json,zipfile
P=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
files=[P/n for n in ['protocol.json','manifest.json','models_frozen.json'] if (P/n).is_file()]
for folder in ['engineering','runs','hog_models','final_results']:
    files.extend(p for p in (P/folder).rglob('*') if p.is_file() and p.suffix not in ['.pt','.pth','.pyc'])
m={'files':{f.relative_to(P).as_posix():sha(f) for f in files},'checkpoints_retained_on_server':{f.relative_to(P).as_posix():sha(f) for f in (P/'runs').rglob('*.pt')}}
(P/'return_manifest.json').write_text(json.dumps(m,indent=2),encoding='utf8');z=P/'point_spv_formal_return.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as out:
    for f in files+[P/'return_manifest.json']:out.write(f,f.relative_to(P))
with zipfile.ZipFile(z) as check:assert check.testzip() is None
z.with_suffix('.sha256.txt').write_text(sha(z)+'  '+z.name+'\n');print(sha(z),z)
