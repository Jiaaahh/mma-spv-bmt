"""G09-A 训练核心：候选化合同 + 与合同逐项一致的优化器。

在项目 `g09_v1/` 内实现，只读复用 v1 `core.py`（资产校验、采样器、BN 冻结、
检查点序列化）。与 v1 的差别**全部显式**记录在新合同的 `overridden_legacy_fields`
与 `candidate` 字段中：学习率、weight_decay、XDM 域头绑定，以及 5000 步 inner 的用途。

工程测试钩子（`engineering=`）仅在 `scope='ENGINEERING_ONLY'` 时可用，用于 CPU 小模型/
桩算法测试；`g09_runner.py` 的 CLI **不暴露**这两个钩子，正式 scope 传入即报错。
"""
from __future__ import annotations

import copy
from decimal import Decimal
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from g09_config import (G09_DIR, HEAD_TO_BACKBONE_RATIO, INNER_DIAGNOSTIC_STEPS, INNER_SELECTION_STEP,
                        PROTOCOL_ID, ROOT, RUNS_DIR_NAME, SAMPLER_ID, SELECTION_RULE_ID, TOTAL_BATCH_PER_CLASS,
                        V1_ANCHOR_CANDIDATE, candidate, expected_param_group_plan, g09_source_hashes,
                        protocol_hashes, sha)

import core as v1_core  # noqa: E402  (g09_config 先把 ROOT 放进 sys.path)

OVERRIDDEN_LEGACY_FIELDS = [
    'optimizer.backbone_lr', 'optimizer.head_lr', 'optimizer.weight_decay',
    'optimizer.domain_lr_candidate', 'xdm_hparams.lr', 'xdm_hparams.weight_decay',
    'inner_selection_step(用途：5000 步 inner 作为唯一候选选择输入)',
]


def build_g09_contract(experiment, cand: dict, run_id: str) -> dict:
    """构造 G09 合同（schema 2）：保留 v1 字段，叠加候选化与 G09 元数据。"""
    legacy = experiment.contract
    hashes = protocol_hashes()
    contract = copy.deepcopy(legacy)
    contract.update({
        'schema': 2,
        'protocol_id': PROTOCOL_ID,
        'candidate_id': cand['candidate_id'],
        'candidate': {
            'backbone_lr': cand['backbone_lr'], 'head_lr': cand['head_lr'],
            'weight_decay': cand['weight_decay'], 'xdm_domain_lr': cand['xdm_domain_lr'],
            'xdm_domain_weight_decay': cand['xdm_domain_weight_decay'],
        },
        'inner_diagnostic_steps': list(INNER_DIAGNOSTIC_STEPS),
        'inner_selection_step': INNER_SELECTION_STEP,
        'selection_rule_id': SELECTION_RULE_ID,
        'optimizer': {
            'name': 'Adam', 'backbone_lr': cand['backbone_lr'], 'head_lr': cand['head_lr'],
            'weight_decay': cand['weight_decay'], 'betas': [0.9, 0.999], 'eps': 1e-8,
            'foreach': False, 'domain_lr_candidate': cand['xdm_domain_lr'],
            'domain_weight_decay_candidate': cand['xdm_domain_weight_decay'],
            'param_group_plan': expected_param_group_plan(cand),
            'weight_decay_semantics': 'Adam L2 on every parameter of every existing group (head, bias, trainable BN affine); no exemption rules',
        },
        'xdm_hparams': copy.deepcopy(experiment.hp),
        'overridden_legacy_fields': list(OVERRIDDEN_LEGACY_FIELDS),
        'g09_source_hashes': g09_source_hashes(),
        'g09_protocol_sha256': hashes['g09_protocol_sha256'],
        'g09_candidate_table_sha256': hashes['g09_candidate_table_sha256'],
        'g09_run_matrix_sha256': hashes['g09_run_matrix_sha256'],
        'g09_handover_sha256': hashes['g09_handover_sha256'],
        'original_manifest_sha256': hashes['original_manifest_sha256'],
        'interpretation': ('G09 finite hyper-parameter search around the executed v1 anchor (C03), not a full optimum claim; '
                           'all candidates restart from step0 of the same ImageNetV1 + fold warm head; the 5000-step inner '
                           'validation is the only candidate-selection input; two fonts remain development data.'),
    })
    contract['xdm_hparams']['lr'] = cand['backbone_lr']
    contract['xdm_hparams']['weight_decay'] = cand['weight_decay']
    return contract


class G09Experiment(v1_core.Experiment):
    """候选化 Experiment：同一份资产校验/采样/冻结 BN 逻辑，换用候选有效优化器。"""

    def __init__(self, method, fold, k, candidate_id, seed=17, device='cpu', threads=4,
                 scope='TRAINING_CANDIDATE', engineering=None):
        if scope == 'TRAINING_CANDIDATE' and engineering:
            raise ValueError('Engineering model/stub hooks are forbidden for TRAINING_CANDIDATE scope')
        if engineering and set(engineering) - {'tiny_net', 'stub_algo', 'stub_domain_head'}:
            raise ValueError(f'Unknown engineering hook(s): {sorted(set(engineering))}')
        if scope == 'TRAINING_CANDIDATE':
            from g09_validation import require_fixed_arguments, verify_source_manifest
            require_fixed_arguments(method, fold, k, seed, threads)
            verify_source_manifest()
        cand = candidate(candidate_id)
        super().__init__(method, fold, k, seed=seed, device=device, threads=threads, scope=scope)
        self.candidate_id = candidate_id
        self.engineering = dict(engineering) if engineering else None
        if self.engineering:
            if method == 'erm' and 'tiny_net' in self.engineering:
                self.net = self.engineering['tiny_net']().to(self.device)
            if method == 'xdm' and 'stub_algo' in self.engineering:
                self.algo = self.engineering['stub_algo'](self.hp, self.device)
                self.net = self.algo.network.network
                self.mix = self.algo.network.mixup_module
        # 候选有效超参：原地更新 hp（作者算法持有同一 dict 引用），再重建两个优化器。
        self.hp['lr'] = cand['backbone_lr']
        self.hp['weight_decay'] = cand['weight_decay']
        if self.algo is not None and getattr(self.algo, 'hparams', None) is not self.hp:
            self.algo.hparams = self.hp
        backbone = [value for name, value in self.net.named_parameters() if not name.startswith('fc.')]
        head = list(self.net.fc.parameters())
        self.optimizer = torch.optim.Adam(
            [{'params': backbone, 'lr': cand['backbone_lr'], 'weight_decay': cand['weight_decay']},
             {'params': head, 'lr': cand['head_lr'], 'weight_decay': cand['weight_decay']}],
            weight_decay=cand['weight_decay'], foreach=False)
        if self.algo is not None:
            self.algo.optimizer = self.optimizer
            self.domain_optimizer = torch.optim.Adam(
                self.algo.network.domain_classifier.parameters(),
                lr=cand['xdm_domain_lr'], weight_decay=cand['xdm_domain_weight_decay'], foreach=False)
            self.algo.domain_optimizer = self.domain_optimizer
        self._guard_optimizer(self.optimizer)
        if self.domain_optimizer is not None:
            self._guard_optimizer(self.domain_optimizer)
        self.freeze_bn()
        self.bn_reference = self.bn_buffers()
        from g09_config import run_id as _run_id
        self.run_id = _run_id(fold, method, candidate_id)
        self.contract = build_g09_contract(self, cand, self.run_id)
        self.verify_optimizer_matches_contract()
        if scope == 'TRAINING_CANDIDATE':
            from g09_validation import validate_contract
            validate_contract(self.contract, method, fold, candidate_id)

    # ---- 运行时与合同的一致性（硬性）----
    def verify_optimizer_matches_contract(self) -> None:
        from g09_validation import verify_adam_groups
        verify_adam_groups(self.optimizer)
        verify_adam_groups(self.domain_optimizer)
        if self.domain_optimizer is not None and len(self.domain_optimizer.param_groups) != 1:
            raise RuntimeError('Domain Adam must contain exactly one param_group')
        contract = self.contract
        cand = contract['candidate']
        plan = contract['optimizer']['param_group_plan']
        if type(self.optimizer) is not torch.optim.Adam:
            raise RuntimeError(f'Optimizer must be torch.optim.Adam (not AdamW/subclass): {type(self.optimizer).__name__}')
        defaults = self.optimizer.defaults
        if (tuple(defaults.get('betas', ())), defaults.get('eps'), defaults.get('foreach')) != ((0.9, 0.999), 1e-8, False):
            raise RuntimeError(f'Optimizer defaults differ from contract: {defaults!r}')
        if float(defaults.get('weight_decay', -1)) != float(cand['weight_decay']):
            raise RuntimeError('Optimizer global weight_decay differs from contract')
        if len(self.optimizer.param_groups) != len(plan):
            raise RuntimeError('param_groups count differs from contract plan')
        backbone_names = {id(v) for n, v in self.net.named_parameters() if not n.startswith('fc.')}
        head_names = {id(v) for v in self.net.fc.parameters()}
        want_sets = [backbone_names, head_names]
        for group, want in zip(self.optimizer.param_groups, plan):
            if float(group['lr']) != float(want['lr']) or group['lr'] != want['lr']:
                raise RuntimeError(f"param_groups[{want['index']}].lr={group['lr']!r} != contract {want['lr']!r}")
            if float(group['weight_decay']) != float(want['weight_decay']) or group['weight_decay'] != want['weight_decay']:
                raise RuntimeError(f"param_groups[{want['index']}].weight_decay={group['weight_decay']!r} != contract {want['weight_decay']!r}")
            if {id(p) for p in group['params']} != want_sets[want['index']]:
                raise RuntimeError(f"param_groups[{want['index']}] parameter set differs from contract role {want['role']!r}")
        if float(self.hp['lr']) != float(cand['backbone_lr']) or float(self.hp['weight_decay']) != float(cand['weight_decay']):
            raise RuntimeError('Live xdm hparams differ from candidate')
        if self.algo is not None:
            live = self.algo.hparams
            if float(live['lr']) != float(cand['backbone_lr']) or float(live['weight_decay']) != float(cand['weight_decay']):
                raise RuntimeError('Author algorithm hparams differ from candidate')
            if self.domain_optimizer is None or type(self.domain_optimizer) is not torch.optim.Adam:
                raise RuntimeError('XDM domain optimizer must be torch.optim.Adam')
            dgroup = self.domain_optimizer.param_groups[0]
            if dgroup['lr'] != cand['xdm_domain_lr'] or dgroup['weight_decay'] != cand['xdm_domain_weight_decay']:
                raise RuntimeError('Domain optimizer lr/weight_decay differ from the bound candidate values')
            if {id(p) for p in dgroup['params']} != {id(p) for p in self.algo.network.domain_classifier.parameters()}:
                raise RuntimeError('Domain optimizer parameter set differs from the author domain classifier')

    # ---- 恢复：先拒绝非 G09 / 非同候选，再走 v1 的严格合同比较 ----
    def restore_g09(self, state) -> None:
        stored = state.get('contract') if isinstance(state, dict) else None
        if not isinstance(stored, dict):
            raise RuntimeError('Checkpoint has no contract; refusing to load')
        if stored.get('protocol_id') != PROTOCOL_ID:
            raise RuntimeError(f"Refusing checkpoint without G09 protocol_id (found {stored.get('protocol_id')!r}); "
                               'legacy v1 / G07 / G08 checkpoints are not reusable')
        if stored.get('candidate_id') != self.candidate_id:
            raise RuntimeError(f"Refusing checkpoint of candidate {stored.get('candidate_id')!r} in run {self.candidate_id!r}")
        if 'run_id' in stored:
            raise RuntimeError('Refusing checkpoint whose contract carries a method-specific run_id '
                               '(contracts must stay symmetric between ERM and XDM)')
        self.restore(state)
        self.verify_optimizer_matches_contract()

    def describe(self) -> dict:
        return {'protocol_id': self.contract['protocol_id'], 'run_id': self.run_id,
                'candidate_id': self.candidate_id, 'method': self.method, 'fold': self.fold,
                'backbone_lr': self.contract['candidate']['backbone_lr'],
                'head_lr': self.contract['candidate']['head_lr'],
                'weight_decay': self.contract['candidate']['weight_decay'],
                'domain_lr': self.contract['candidate']['xdm_domain_lr'],
                'domain_weight_decay': self.contract['candidate']['xdm_domain_weight_decay']}


def assert_head_ratio(cand: dict) -> None:
    """名义值比较用十进制：20×1e-6 在二进制浮点里不等于 2e-5，但设计上是 20 倍关系。"""
    head, backbone = Decimal(str(cand['head_lr'])), Decimal(str(cand['backbone_lr']))
    if head != HEAD_TO_BACKBONE_RATIO * backbone:
        raise RuntimeError(f"Candidate {cand['candidate_id']}: head_lr must be exactly {HEAD_TO_BACKBONE_RATIO}x backbone_lr")
    if Decimal(str(cand['xdm_domain_lr'])) != backbone or \
            Decimal(str(cand['xdm_domain_weight_decay'])) != Decimal(str(cand['weight_decay'])):
        raise RuntimeError(f"Candidate {cand['candidate_id']}: domain head must be bound to the shared candidate values")


def assert_c03_is_v1_anchor(cand: dict) -> None:
    if cand['candidate_id'] != V1_ANCHOR_CANDIDATE:
        raise RuntimeError('Only C03 may be checked as the v1 anchor')
    observed = tuple(Decimal(str(cand[key])) for key in
                     ('backbone_lr', 'head_lr', 'weight_decay', 'xdm_domain_lr', 'xdm_domain_weight_decay'))
    if observed != tuple(Decimal(str(v)) for v in (5e-6, 1e-4, 0.0, 5e-6, 0.0)):
        raise RuntimeError('C03 must equal the executed v1 anchor (5e-6 / 1e-4 / 0 / 5e-6 / 0)')
