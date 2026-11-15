from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
def read(p): return json.loads((OUT/p).read_text(encoding='utf-8'))
def save(p,x): (OUT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256((OUT/p).read_bytes()).hexdigest()
b=read('runtime_benchmark.json'); source=read('clip_source_manifest.json')
assert b['status']=='PASS' and read('text_audit.json')['status']=='PASS' and read('math_audit.json')['status']=='PASS'
assert len(read('installed_source_audit.json')['files'])==5 and all(read('installed_source_audit.json')['files'].values())
save('visual_input_audit.json',dict(status='PASS',review='Assistant viewed training_input_preview.png; six glyphs across ideal/2026/2029 mappings show visible phosphene structure and complete square frames; no unintended reflection or all-black quantization.',
    preview_sha256=sha('training_input_preview.png'),scope='18 training images; not a universal input correctness or recognition test',
    normalization='per-image maximum, removes absolute brightness; task adaptation disclosed',formal_accuracy_computed=False))
protocol=dict(status='FROZEN_BEFORE_NEW_ACCURACY',timestamp=datetime.now(timezone.utc).isoformat(),
    identity='Full-core training-free Proto-Adapter adapted to SPV Chinese characters, paper-derived implementation; not author-code reproduction',
    source_commit=source['commit'],weight_sha256=source['weight_sha256'],backbone='OpenAI CLIP RN50',
    frozen_image_and_text=True,gradient_updates=False,image_dim=1024,device='cpu',threads=2,batch=8,
    image_conversion=b['conversion'],resize=224,interpolation='bicubic',crop='224 center crop after resizing 96x96 square to 224x224; removes no spatial region',
    augmentations='No flips or random crops; mapping views retained',prompt_templates=read('prompts.json')['templates'],
    text_ensemble='L2 each template vector, average seven, then L2',
    prototype='L2 support samples; per-class mean; column L2 across classes; row L2 across channels',
    query='sample L2 only',score='alpha*(q@P.T)+(q@T.T)',
    alpha_grid=[0]+[10**(-6+i*.25) for i in range(49)],alpha_selection='PA0 and PA1 independently maximize all 5400 DEV query accuracy; ties smallest alpha',
    boundary_rule='If best nonzero alpha is smallest/largest positive grid value, extend that edge by four decades once with step .25, using DEV only; log complete search',
    train_maps=['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)],dev_maps=[f'moderate_seed{s}' for s in range(5301,5306)],
    folds='6 leave-one-font-out; fit prototypes and channel normalization on five support fonts only',
    primary='E1 H1 accuracy minus PA1 accuracy',secondary='C1 same paired comparison',panels='historical, not new blinded panels',
    bootstrap=dict(unit='character cluster',draws=100000,E1_seed=26092932,C1_seed=26092933),
    diagnostics=['text only','PA0 visual only','PA1 visual only','visual prototype without column normalization'],
    result_retention='All valid outcomes retained, including losses; invalid runs documented separately',
    precision='CLIP inference float32; subsequent prototype aggregation and scoring float64',
    implementation_contract='Future scoring runner must match this protocol and pass input hashes, fold disjointness, label order, and math checks before scoring',
    evidence_hashes={n:sha(n) for n in ['第一步核对与比较协议.md','classes.json','prompts.json','prompt_tokens.npy','text_features.npy','requirements_locked.txt','runtime_preflight.py','input_audit.json']})
save('frozen_protocol.json',protocol)
save('readiness.json',dict(stage='step_2_local_runtime_preflight',status='PASS_READY_FOR_FEATURE_EXTRACTION',
    runtime_gates_passed=True,new_recognition_accuracy_computed=False,manuscripts_modified=False,
    frozen_protocol_sha256=sha('frozen_protocol.json'),
    measured_images_per_second=b['images_per_second'],projected_image_minutes=60*b['projected_16200_image_hours'],
    remaining_work=['Implement and audit full scoring runner against frozen protocol','Extract and cache full image features','DEV-only fusion selection, then fixed E1/C1 scoring','Paired statistics and evidence crosswalk'],
    scope_note='This stage ends after local readiness and benchmark. Full image extraction and formal scoring have not started.'))
with (OUT/'evidence_crosswalk.csv').open('a',encoding='utf-8') as f:
    f.write('source_and_weight,clip_source_manifest.json,PASS,pending\ninstalled_source,installed_source_audit.json,PASS,pending\ntext_encoder,text_audit.json,PASS,pending\nmath,math_audit.json,PASS,pending\ninput_preview,visual_input_audit.json,PASS,pending\nruntime,runtime_benchmark.json,PASS,pending\nfrozen_protocol,frozen_protocol.json,FROZEN_BEFORE_NEW_ACCURACY,pending\n')
save('step2_delivery_manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='step2_delivery_manifest.json'})
print('LOCAL PREFLIGHT COMPLETE; NO ACCURACY COMPUTED')
