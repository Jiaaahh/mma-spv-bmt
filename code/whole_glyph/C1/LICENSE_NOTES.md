# License and provenance notes

This project is a new experimental workspace derived from the local
`hanzi-spv-benchmark` implementation.  The original frozen repository remains
read-only and is not modified by this project.

The development runner reads the existing glyph and mapping archives from the
path configured as `legacy_root`.  Those data files and font files are not
redistributed in this project.  Their original manifests, font licenses, and
provenance records remain under the legacy repository.

Before sharing the project outside the current computer, review the upstream
source-code license and every referenced font license.  Do not bundle the
legacy data or fonts unless their licenses and provenance records are included.

The C1 confirmation assets were rendered locally from DengXian Regular
(`/historical/Windows\Fonts\Deng.ttf`) and FZShuTi Regular
(`/historical/Windows\Fonts\FZSTK.TTF`).  The raw font files are not included.  The
project stores only the resulting 96x96 glyph archive, contact sheets, quality
control metadata, and cryptographic hashes.  Do not add or redistribute the raw
font files without separately confirming their licenses.
