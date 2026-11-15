# Hanzi-SPV MMA extension E1

This sibling project contains the supplemental experiments for the positive
multi-mapping augmentation result.  It does not modify or overwrite the
completed C1 project in `../hanzi-spv-calibrated-v2`.

## What E1 adds

1. Four additional test fonts (720 glyphs) with blind technical QC.
2. Five paired mapping conditions and five new latent seeds (25 frozen
   condition-seed mappings; conditions derived from the same seed share its
   latent draw and are not 25 independent random mappings).
3. MMA dose and ideal-anchor ablations.
4. A balanced LinearSVC decoder baseline, with C selected only on D3.
5. A Scialom-inspired Gabor/clock-fragment mechanism control adapted to the
   fixed 16 x 16 grid and compared under the same renderer/classifier exposure;
   it is neither an exact replication nor a state-of-the-art baseline for this
   task.
6. Character-cluster paired bootstrap, Macro-F1, and efficiency reporting.

## Ordered workflow

```powershell
python prepare_e1_assets.py
python run_e1_development.py --config configs/d3_svc.yaml
python freeze_e1_confirmation.py --config configs/e1_confirmation.yaml
python freeze_e1_confirmation.py --config configs/e1_confirmation.yaml --validate
python run_e1_confirmation.py --config configs/e1_confirmation.yaml
python run_e1_analysis.py --config configs/e1_confirmation.yaml
python -m pytest -q
```

The checked-in local E1 assets were generated before any E1 prediction.  The
formal confirmation runner refuses to overwrite an existing completed output.
The analysis command is separate so no aggregate accuracy is exposed until all
25 condition/seed cells are committed.

## Important interpretation boundary

The Scialom baseline is a declared fixed-grid adaptation, not a claim of exact
replication of the original continuous-position, human object-recognition
experiment.  The four local system fonts and their derived glyph archive also
must not be redistributed without separate permission.  A public replay should
substitute openly licensed fonts and create a new frozen confirmation study.

See `docs/E1_PREREGISTRATION.md` for the hypotheses, comparisons, and stopping
rule specified before performance access and protected by the internal hash
lock.  This was an internal prospective protocol, not a public preregistration.
