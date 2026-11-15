"""No-training A2 regression suite. Synthetic logits only; no outer scores are read."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import numpy as np
import torch
import g09_config as cfg
import g09_validation as validation
import g09_select as selector
import g09_gate as gate
import g09_compare as compare
import g09_runner as runner
import g09_core as core
import fixtures

class A2Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        validation.verify_source_manifest()
        cls.root = cfg.ROOT/'engineering_g09_v1'/('a2_'+uuid.uuid4().hex)
        cls.selections = cls.root/'selections_g09_v1'
        cls.box = dict(root=cls.root,runs=cls.root,selections=cls.selections)
        cls.root.mkdir(parents=True)
        # Use frozen formal contract metadata; checkpoints remain explicitly fake bytes.
        def fixture_contract(method, fold, candidate_id, **overrides):
            value = validation.expected_contract(method,fold,candidate_id)
            value.update(overrides)
            return value
        with patch.object(fixtures,'fixture_contract',fixture_contract):
            for method in cfg.EXPECTED_METHODS:
                for fold in cfg.EXPECTED_FOLDS:
                    for i,cid in enumerate(cfg.EXPECTED_CANDIDATES):
                        fixtures.make_run(cls.box,fold,method,cid,correct_target=400-i)
                    selector.select(method,fold,run_root=cls.root,out_dir=cls.selections)

    def allowed(self):
        return gate.require_outer_allowed('erm',0,cfg.run_path(0,'erm','C01'),self.selections,self.root)

    def mutation_rejected(self, mutate, method='erm', fold=0, reseal=True):
        path=gate.selection_path(method,fold,self.selections)
        original=path.read_bytes()
        data=json.loads(original)
        mutate(data)
        if reseal:
            data['integrity_sha256']=gate.selection_digest(data)
        cfg.write_json(path,data)
        try:
            with self.assertRaises((RuntimeError,ValueError,KeyError,TypeError)):
                self.allowed()
        finally:
            path.write_bytes(original)

    def test_valid_all_four_and_winner(self):
        self.assertEqual(self.allowed()['winner']['candidate_id'],'C01')

    def test_empty_input_rejected(self):
        self.mutation_rejected(lambda d:d.update(input_hashes={},required_candidates=0))

    def test_missing_input_rejected(self):
        self.mutation_rejected(lambda d:d.pop('input_hashes'))

    def test_wrong_candidate_set(self):
        def change(d): d['input_hashes']['C99']=d['input_hashes'].pop('C06')
        self.mutation_rejected(change)

    def test_winner_replacement_even_with_new_digest(self):
        self.mutation_rejected(lambda d:d['winner'].update(candidate_id='C06',run_path=cfg.run_path(0,'erm','C06')))

    def test_wrong_ranking_even_with_new_digest(self):
        self.mutation_rejected(lambda d:d['ranking'].reverse())

    def test_identity_mismatch(self):
        self.mutation_rejected(lambda d:d.update(fold=1))

    def test_integrity_digest(self):
        self.mutation_rejected(lambda d:d.update(note='changed'),reseal=False)

    def test_other_fold_input_change(self):
        path=self.root/cfg.run_path(1,'xdm','C06')/'latest.pt'
        original=path.read_bytes()
        try:
            path.write_bytes(original+b'changed')
            with self.assertRaises(RuntimeError): self.allowed()
        finally: path.write_bytes(original)

    def test_loser_outer_rejected(self):
        with self.assertRaises(RuntimeError):
            gate.require_outer_allowed('erm',0,cfg.run_path(0,'erm','C06'),self.selections,self.root)

    def test_outer_directory_does_not_deadlock_frozen_revalidation(self):
        run=self.root/cfg.run_path(0,'erm','C01')
        directory=run/'evaluations/outer_step5000_A2_EMPTY_PLACEHOLDER'
        directory.mkdir()
        try:
            self.allowed() # Reads inner only; empty outer directory is never opened.
            with self.assertRaises(RuntimeError):
                selector.read_inner(cfg.run_path(0,'erm','C01'),'erm',0,'C01',run_root=self.root)
        finally: directory.rmdir()

    def test_valid_different_candidates(self):
        a=validation.expected_contract('erm',0,'C01')
        b=validation.expected_contract('xdm',0,'C02')
        self.assertFalse(compare.declared_differences(a,b)['same_candidate'])

    def test_undeclared_changes_and_shared_drift(self):
        changes=[('xdm_hparams','warmup_step',1),('xdm_hparams','threshold',.7),
                 ('optimizer','betas',[.1,.2]),('optimizer','eps',1e-4),('optimizer','foreach',True)]
        for section,key,value in changes:
            for both in (False,True):
                with self.subTest(section=section,key=key,both=both):
                    a=validation.expected_contract('erm',0,'C01')
                    b=validation.expected_contract('xdm',0,'C02')
                    b[section][key]=value
                    if both: a[section][key]=value
                    with self.assertRaises(RuntimeError): compare.declared_differences(a,b)

    def test_fixed_arguments_before_model_or_files(self):
        from types import SimpleNamespace
        for k,threads in ((4,4),(8,1)):
            args=SimpleNamespace(method='erm',fold=0,k=k,seed=17,threads=threads)
            with patch.object(runner,'resolve_declared_run',side_effect=AssertionError('filesystem stage reached')):
                with self.assertRaises(ValueError): runner.train(args)
            with patch.object(core.v1_core.Experiment,'__init__',side_effect=AssertionError('base init reached')):
                with self.assertRaises(ValueError): core.G09Experiment('erm',0,k,'C01',threads=threads)

    def test_contract_fixed_values_and_scope(self):
        for key,value in [('k',4),('seed',18),('total_batch',20),('scope','ENGINEERING_ONLY')]:
            contract=validation.expected_contract('erm',0,'C01');contract[key]=value
            with self.assertRaises(RuntimeError): validation.validate_contract(contract,'erm',0,'C01')

    def test_cli_rejects_k_and_threads_before_dispatch(self):
        import contextlib
        import io
        for flag,value in [('--k','4'),('--threads','1')]:
            with patch.object(runner,'dispatch',side_effect=AssertionError('dispatch reached')):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        runner.main(['train','--method','erm','--fold','0','--candidate','C01',
                            '--device','cuda','--until','1','--run',cfg.run_path(0,'erm','C01'),flag,value])
                    self.assertEqual(error.exception.code,2)

    def test_run_matrix_identity_mismatch(self):
        with self.assertRaises(RuntimeError):
            runner.resolve_declared_run('erm',0,'C01',cfg.run_path(0,'erm','C02'))

    def test_missing_one_selection_blocks(self):
        path=gate.selection_path('xdm',1,self.selections)
        original=path.read_bytes()
        path.unlink()
        try:
            with self.assertRaises(RuntimeError): self.allowed()
        finally: path.write_bytes(original)

    def test_live_core_verifier_checks_loaded_groups(self):
        from types import SimpleNamespace
        from engineering_kit import TinyNet
        net=TinyNet()
        contract=validation.expected_contract('erm',0,'C01')
        spec=contract['candidate']
        optimizer=torch.optim.Adam([
            {'params':[p for n,p in net.named_parameters() if not n.startswith('fc.')],'lr':spec['backbone_lr']},
            {'params':net.fc.parameters(),'lr':spec['head_lr']}],weight_decay=spec['weight_decay'],foreach=False)
        fake=SimpleNamespace(optimizer=optimizer,domain_optimizer=None,contract=contract,net=net,
                             hp=contract['xdm_hparams'],algo=None)
        core.G09Experiment.verify_optimizer_matches_contract(fake)
        state=optimizer.state_dict();state['param_groups'][1]['eps']=1e-4
        optimizer.load_state_dict(state)
        with self.assertRaises(RuntimeError): core.G09Experiment.verify_optimizer_matches_contract(fake)

    def test_effective_adam_after_state_load(self):
        p=torch.nn.Parameter(torch.zeros(1))
        for key,value in [('betas',(.1,.2)),('eps',1e-4),('foreach',True),('amsgrad',True)]:
            optimizer=torch.optim.Adam([p],foreach=False)
            state=optimizer.state_dict();state['param_groups'][0][key]=value
            optimizer.load_state_dict(state)
            with self.assertRaises(RuntimeError): validation.verify_adam_groups(optimizer)
        validation.verify_adam_groups(torch.optim.Adam([p],foreach=False))

    def test_source_manifest_stale(self):
        path=cfg.G09_DIR/'G09_PACKAGE_MANIFEST.json'
        original=path.read_bytes()
        data=json.loads(original);data['new_source_files']['g09_v1/make_g09_a_package.py']='0'*64
        try:
            cfg.write_json(path,data)
            with self.assertRaises(RuntimeError): validation.verify_source_manifest()
        finally: path.write_bytes(original)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(A2Regression)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'suite':'G09_A2_REGRESSION','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
        'notice':'Synthetic fixtures only; no model training or real outer evaluation.',
        'source_hashes':cfg.g09_source_hashes()}
    cfg.write_json(cfg.G09_DIR/'evidence_a2/regression.json',report)
    sys.exit(0 if result.wasSuccessful() else 1)
