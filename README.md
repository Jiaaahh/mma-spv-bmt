# MMA-SPV / BMT reproducibility materials

Version **1.0.0** · [Repository](https://github.com/Jiaaahh/mma-spv-bmt)

Historical source code, saved numerical predictions, figure/table source numbers, selected historical runtime arrays, and the E01–E32 evidence map for a simulated prosthetic-vision study of multi-mapping prototype aggregation. The associated 20261114 BMT manuscript and supplement are identified by hashes in `index/manuscript_binding.json`; manuscript files are not included.

## Get started

Download or clone this repository. Code and numerical evidence are included together; no separate data archive is needed for the commands below. See [RUN_PUBLIC.md](RUN_PUBLIC.md) for all numerical checks and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for their limits.

```sh
python -X utf8 -B tools/check_public_release.py --output-dir checks/public_integrity
python -X utf8 -B tools/verify_results.py --bundle-root evidence --output-dir checks/public_predictions
python -X utf8 -B tools/export_main_results.py --output-dir checks/public_main_results
```

Python 3.11+ is required by the integrity entry. Prediction recount/export additionally requires NumPy. Statistical replays use the recorded scientific environment in `environment/requirements-step3-inference.txt`; these direct requirements are not a complete transitive dependency lock. Supported commands write generated output under `checks/`.

## Scope and limitations

This distribution supports **numerical verification from saved predictions**. It excludes raw fonts, frozen glyph bitmap caches, per-glyph stimulation/selection arrays, reconstructed prototypes, figure images, neural checkpoints and pretrained weights. It cannot provide complete end-to-end model inference or pixel-identical figure regeneration.

The complete controlled Step 3 package replayed **91,800 MMA predictions** and **10 figure PNGs**. These are historical local audit results, not executions using this reduced public distribution. The six original Point-SPV checkpoints are unavailable; the four selected G09 checkpoints remain unverified. Missing Point vendor modules are listed in `index/missing_point_vendor.json`.

The E01–E32 maps retain negative results and the distinctions between development, formal confirmation, post-hoc work, whole-glyph and common-window protocols. Original computational and scientific conclusions are not expanded by this release. Other models and all 36 supplementary tables have not been comprehensively rerun, and historical runtime-array checks are not new hardware measurements.

Public paths are redacted. Retained NPZ arrays match the controlled originals; excluded keys are recorded in `provenance/array_transformations.json`. Source-to-candidate hashes, candidate-to-public changes and the final distribution manifest record which bytes were retained or changed. Embedded historical locks/manifests retain their original meaning; `manifest.json` binds the actual public distribution.

## License and attribution

Author-controlled source: **Apache-2.0** ([LICENSE](LICENSE)). Author-controlled numerical evidence and documentation: **CC BY 4.0** ([LICENSE-DATA.md](LICENSE-DATA.md)). Third-party terms remain separate; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Maintainer account: [Jiaaahh](https://github.com/Jiaaahh). Formal scholarly creator metadata has not yet been supplied. See [CITATION.md](CITATION.md); no author names, affiliations or ORCIDs are inferred. Zenodo archiving and DOI registration have not been performed.

## Verification record

The preceding local candidate passed nine numerical/integrity entries. Its detailed execution sequence and limitations are preserved in `validation/step4_candidate_validation.json`. The publication changes are documentation, licenses and repository-integrity handling; retained computation and numerical inputs are unchanged. See `validation/github_release_checks.json` for the checks actually repeated for this publication.

Codex assistance and the workflow source are recorded in [PROVENANCE.md](PROVENANCE.md), without adding manuscript references.
