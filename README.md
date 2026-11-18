# NISAR–GEDI Mangrove AGBD Reproducibility Code

Code and supporting material for the manuscript:

**Conditional Association of Provisional NISAR L-Band GCOV Predictors with GEDI-Derived Mangrove Biomass in the Bangladesh Sundarbans: A Spatially Structured Evaluation**

## Authors

1. Imran Ahmed — Beihang University, Hangzhou International Campus, Hangzhou 311115, Zhejiang, China — ORCID 0009-0004-6010-620X
2. Sheikh Tawhidul Islam — Regional Centre for Space Science and Technology Education in Asia and the Pacific (RCSSTEAP), Beihang University, Hangzhou 311115, Zhejiang, China — ORCID 0000-0002-9697-4306
3. Md. Sarafat Islam — University of Science and Technology of China (USTC), Hefei 230026, Anhui, China — ORCID 0009-0006-5661-9410
4. Yue Zhang — Beihang University, Hangzhou International Campus, Hangzhou 311115, Zhejiang, China — ORCID 0009-0000-6723-5822

Primary corresponding author: Yue Zhang (blairz@buaa.edu.cn)  
Second corresponding author: Sheikh Tawhidul Islam (s.t.islam@juniv.edu)

## Repository design

`code/executed_original/` preserves the supplied original analysis scripts from the authors' Windows workspaces.

`code/portable/` contains minimally cleaned copies for public reuse. Path configuration was made portable through environment variables and interactive terminal pauses were removed. The analytical workflow and model logic were not intentionally altered.

`reviewer_validation/` contains the frozen major-revision validation package and supporting reproducibility materials. The Rev6.4 submission additionally uses the versioned `Rev64_Review_Supplement_v1.1.0.zip` package prepared from the authoritative project tree.

See `docs/RUN_ORDER.md` for the original workflow and `reviewer_validation/README.md` for the major-revision validation package.

## Scientific design

- NISAR L2 GCOV PROVISIONAL as predictor source
- GEDI L4A Version 3 as reference AGBD
- 1,641 final matched GEDI–NISAR observations
- 3-km spatial block tessellation
- five geographic outer folds
- 500-m fold-specific training exclusion
- multiscale NISAR predictor evaluation
- separate June 25, 2026 temporal-sensitivity experiment
- C15 alternative smooth-coordinate spatial-baseline sensitivity
- explicit tests of geographic transferability, temporal mismatch, GEDI uncertainty, predictor-support scale, mangrove-edge mixing, conditional NISAR information, and dependence-unit sensitivity

The study is a sensor-sensitivity and methodological evaluation. It does not present the fitted model as an operational wall-to-wall biomass estimator.

## Reproducibility archives

### Historical immutable release
**Release:** v1.0.2  
**Zenodo DOI:** 10.5281/zenodo.23740025

The v1.0.2 tag/release remains immutable and is not overwritten.

### Rev6.4 submission package
**Planned release:** v1.1.0  
**Package:** `Rev64_Review_Supplement_v1.1.0.zip`

The v1.1.0 review/submission supplement contains the input-granule manifest, software-environment record, feature-extraction and validation scripts, frozen fold memberships, primary and conditional footprint-level out-of-fold predictions, C15 sensitivity outputs, sample-flow/attrition summaries, diagnostic figures, regeneration guidance, and SHA-256 checksums. A permanent Zenodo DOI will be added after the v1.1.0 archive is published.

Raw NISAR, GEDI, and Global Mangrove Watch source data are not redistributed.

## Data availability

Raw NISAR, GEDI, and Global Mangrove Watch products must be obtained from their original data providers.

This repository and the versioned reproducibility archive provide analysis code, frozen validation memberships, OOF predictions, hyperparameters, QC results, source-product identifiers, environmental information, diagnostics, and checksums required to reproduce the reported workflow once the source datasets are obtained.

## License

MIT
