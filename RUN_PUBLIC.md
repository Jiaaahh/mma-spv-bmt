# Run the public numerical checks

Clone or download this repository, then run from its root. Code and numerical evidence are included together. Commands write new outputs below `checks/`; they do not train or start original one-shot confirmation runners. `checks/` is generated output, not a public source input.

```sh
python -X utf8 -B tools/prepare_data.py
python -X utf8 -B tools/check_public_release.py --output-dir checks/public_integrity
python -X utf8 -B tools/verify_results.py --bundle-root evidence --output-dir checks/public_predictions
python -X utf8 -B tools/export_main_results.py --output-dir checks/public_main_results
python -X utf8 -B tools/recompute_primary_statistics.py --panel C1 --output-dir checks/public_primary_C1
python -X utf8 -B tools/recompute_primary_statistics.py --panel E1 --output-dir checks/public_primary_E1
python -X utf8 -B tools/recompute_primary_statistics.py --panel StudyB --output-dir checks/public_primary_StudyB
python -X utf8 -B tools/recompute_comparators.py --output-dir checks/public_comparators
python -X utf8 -B tools/recompute_point_and_costs.py --output-dir checks/public_point_and_costs
python -X utf8 -B tools/recompute_sensitivity.py --bundle-root evidence --output-dir checks/public_sensitivity
```

Use the recorded Python/scientific versions for interval replays. Original files embedded within `evidence/` and `code/` include historical paths and historical hash expectations after redaction; they are not all portable entry points. The `fetch_clip.py` historical helper has been pinned to the recorded CLIP commit but has not been executed in this release and is not needed for the numerical checks.

`tools/replay_mma.py` and figure-rendering commands are excluded because required rights-restricted inputs are absent. Original font identities and hashes are in `evidence/font_identity.csv`; they are identification records, not redistribution permissions.

Large prediction CSVs are stored as lossless gzip for reliable transport. `tools/prepare_data.py` restores their original paths only after checking both archive and original-file SHA-256. The integrity entry also performs this preparation automatically. Numerical predictions and historical evidence hashes are unchanged.
