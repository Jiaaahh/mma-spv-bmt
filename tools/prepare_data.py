"""Restore losslessly compressed evidence and verify the original SHA-256."""
from pathlib import Path
import gzip,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def materialize():
    records=json.loads((ROOT/'provenance/compressed_data.json').read_text(encoding='utf-8'))['files']
    for item in records:
        target=(ROOT/item['path']).resolve();archive=(ROOT/item['archive']).resolve()
        if not target.is_relative_to(ROOT/'evidence') or not archive.is_relative_to(ROOT/'evidence'):raise ValueError('Data path escapes evidence/')
        if target.exists():
            if target.stat().st_size!=item['bytes'] or sha(target)!=item['sha256']:raise ValueError('Existing evidence hash mismatch: '+item['path'])
            continue
        if sha(archive)!=item['archive_sha256']:raise ValueError('Compressed evidence hash mismatch')
        tmp=target.with_suffix(target.suffix+'.partial')
        if tmp.exists():raise ValueError('Incomplete prior extraction exists: '+tmp.name)
        try:
            total=0
            with gzip.open(archive,'rb') as src,tmp.open('xb') as out:
                while chunk:=src.read(1024*1024):
                    total+=len(chunk)
                    if total>item['bytes']:raise ValueError('Unexpected decompressed size')
                    out.write(chunk)
            if total!=item['bytes'] or sha(tmp)!=item['sha256']:raise ValueError('Restored evidence hash mismatch')
            tmp.replace(target)
        except Exception:
            if tmp.exists():tmp.unlink()
            raise
    return len(records)
if __name__=='__main__':print(json.dumps({'passed':True,'compressed_evidence_files_ready':materialize()}))
