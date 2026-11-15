"""Check this public distribution's bytes and evidence map; do not execute historical CLIs."""
from pathlib import Path, PurePosixPath
import argparse, ast, csv, hashlib, json, re

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve()
    if (ROOT/'checks').resolve() not in out.parents:raise ValueError('Output must be inside checks/')
    from prepare_data import materialize
    materialize()
    failures=[];m=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));seen=set()
    for item in m['files']:
        name=item['path'];path=PurePosixPath(name)
        if name in seen or path.is_absolute() or '..' in path.parts:
            failures.append({'file':name,'problem':'unsafe_or_duplicate'});continue
        seen.add(name);p=ROOT/name
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:failures.append({'file':name,'problem':'hash_mismatch_or_missing'})
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not {'checks','__pycache__','.git'}.intersection(p.relative_to(ROOT).parts) and p!=ROOT/'manifest.json'}
    generated={x['path'] for x in m.get('generated_files',[])}
    actual-=generated
    if actual!=seen:failures.append({'problem':'unexpected_or_unlisted_files','extra':sorted(actual-seen),'missing':sorted(seen-actual)})
    rows=list(csv.DictReader((ROOT/'index/experiment_commands.csv').open(encoding='utf-8-sig',newline='')))
    if sorted(x['id'] for x in rows)!=[f'E{i:02}' for i in range(1,33)]:failures.append({'problem':'experiment_coverage'})
    sources=0;scanned=0
    patterns=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{30,}',r'sk-[A-Za-z0-9_-]{32,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'\u5f00\u5fc3\u5c31\u597d',r'[A-Za-z]:[/\\]+Users[/\\]+']
    for name in seen:
        p=ROOT/name
        if p.suffix.lower() in {'.ttf','.otf','.ttc','.pt','.pth','.png','.jpg','.jpeg','.pdf','.drawio','.svg'}:failures.append({'file':name,'problem':'excluded_asset_type'})
        if p.suffix.lower() in {'.py','.json','.md','.csv','.yaml','.txt','.toml','.sh'}:
            s=p.read_text(encoding='utf-8-sig');scanned+=1
            if any(re.search(pat,s) for pat in patterns):failures.append({'file':name,'problem':'private_path_or_high_confidence_secret_pattern'})
            if p.suffix=='.py':
                try:ast.parse(s);sources+=1
                except SyntaxError:failures.append({'file':name,'problem':'python_syntax'})
    result={'passed':not failures,'manifest_entries':len(seen),'experiments':len(rows),'python_sources_parsed':sources,'text_files_scanned':scanned,'failures':failures,'scope':'Candidate bytes, syntax, selected private-path/secret patterns and experiment identifiers only; not a legal or complete privacy certification.'}
    out.mkdir(parents=True,exist_ok=True);(out/'integrity_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
