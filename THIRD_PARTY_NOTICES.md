# Third-party provenance and notices

Accessed 2026-11-15. The project license covers only material controlled by the authors; it does not relicense dependencies, fonts, pretrained models or other third-party material.

| Component | Bound source | Terms and inclusion |
|---|---|---|
| Point-SPV | arnejad/E2E-Point-SPV, commit 752af482dd6104d14bd61e794a15738271640c88 | Apache-2.0, text in LICENSES/Point-SPV-Apache-2.0.txt. Historical adaptation imports original Representer. Vendor model modules were absent in the Step 3 input and are not invented here. |
| torchvision ResNet | torchvision 0.21.0 | BSD-3-Clause, LICENSES/torchvision-BSD-3-Clause.txt. Historical provenance records describe extraction; missing vendor source prevents this candidate from replaying Point. |
| CLIP | openai/CLIP, commit d05afc436d78f1c48dc0dbf8e5980a9d471f35f6 | MIT, LICENSES/CLIP-MIT.txt. CLIP source repository and RN50 weights are not bundled. Historical fetch helper is pinned to the recorded commit. |
| RanDumb / Proto-Adapter / mechanism controls | Historical experiment descriptions and code | Paper-derived/adapted implementations retain their stated status; no claim of exact author-code reproduction. Source authorship and rights require the project rights holder's final review. |
| Scientific Python and graphics runtimes | environment/ | Dependencies are installed separately, not vendored here. Preserve their notices when separately redistributing them. |
| Fonts and derived glyph sets | evidence/font_identity.csv and asset manifests | No font binary, frozen glyph bitmap cache, per-glyph stimulation array or figure image is redistributed in this candidate. No grant for these inputs is implied. |

Upstream license sources:

- https://github.com/arnejad/E2E-Point-SPV/blob/752af482dd6104d14bd61e794a15738271640c88/LICENSE
- https://github.com/pytorch/vision/blob/v0.21.0/LICENSE
- https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE

Candidate transformations are recorded per file in provenance/source_to_candidate.json. They include historical path redaction, removal of specified visual-array keys, a rebuilt candidate evidence manifest, and the explicit CLIP commit pin. Retained numerical arrays are unchanged.

The font release decision is a conservative packaging boundary, not a legal conclusion that every derived scalar or image is restricted. The Microsoft FAQ distinguishes text graphics from individual-character bitmap conversions: https://learn.microsoft.com/en-us/typography/fonts/font-faq . Other foundries and font versions require their own terms.
