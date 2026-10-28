# NISAR–GEDI Mangrove AGBD Reproducibility Code

Code supporting the manuscript:

**Early Evaluation of NISAR L-Band Sensitivity to Mangrove Aboveground Biomass Using GEDI and Spatial Machine Learning**

## Authors

- Imran Ahmed — ORCID 0009-0004-6010-620X
- Md. Sarafat Islam — ORCID 0009-0006-5661-9410
- Sheikh Tawhidul Islam — ORCID 0000-0002-9697-4306

Corresponding author: Sheikh Tawhidul Islam

## Repository design

`code/executed_original/` preserves the supplied original analysis scripts from the authors' Windows workspaces.

`code/portable/` contains minimally cleaned copies for public reuse. Path configuration was made portable through environment variables and interactive terminal pauses were removed. The analytical workflow and model logic were not intentionally altered.

`reviewer_validation/` contains the frozen major-revision validation package covering C1–C6 plus the C8 reproducibility audit, including exact fold memberships, OOF predictions, model settings, sensitivity analyses, QC tables, and SHA256 checksums.

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
- explicit tests of geographic transferability, temporal mismatch, GEDI uncertainty, predictor-support scale, and mangrove-edge mixing

The study is a sensor-sensitivity and methodological evaluation. It does not present the fitted model as an operational wall-to-wall biomass estimator.

## Reproducibility archive

**Release:** v1.0.2

**Zenodo DOI:** 10.5281/zenodo.23740025

The v1.0.2 archive supersedes v1.0.1 for reproducibility of the revised manuscript because it includes the complete reviewer-validation package.

The frozen reviewer-validation package contains 463 files, with 462 entries in its SHA256 manifest. Raw NISAR, GEDI, and Global Mangrove Watch source data are not redistributed.

Exact source-product identifiers and acquisition information are provided in:

`reviewer_validation/docs/SOURCE_PRODUCTS.md`

## Data availability

Raw NISAR, GEDI, and Global Mangrove Watch products must be obtained from their original data providers.

This repository provides the analysis code, frozen validation memberships, OOF predictions, hyperparameters, QC results, environmental information, source-product identifiers, and checksums required to reproduce the reported workflow once the source datasets are obtained.

## License

MIT
