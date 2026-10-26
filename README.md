# NISAR–GEDI Mangrove AGBD Reproducibility Code

Code supporting the manuscript:

**Early Evaluation of NISAR L-Band Sensitivity to Mangrove Aboveground Biomass Using GEDI and Spatial Machine Learning**

## Authors

- Imran Ahmed — ORCID 0009-0004-6010-620X
- Md. Sarafat Islam — ORCID 0009-0006-5661-9410
- Sheikh Tawhidul Islam — ORCID 0000-0002-9697-4306

Corresponding author: Sheikh Tawhidul Islam

## Repository design

`code/executed_original/` preserves the supplied analysis scripts exactly as received from the authors' Windows workspaces.

`code/portable/` contains minimally cleaned copies for public reuse. Path configuration was made portable through environment variables and interactive terminal pauses were removed. The analytical workflow and model logic were not intentionally altered.

See `docs/RUN_ORDER.md` for the execution sequence and `docs/CODE_AUDIT.md` for the audit summary.

## Scientific design

- NISAR L2 GCOV PROVISIONAL as predictor source
- GEDI L4A Version 3 as reference AGBD
- 1,641 final matched footprints
- 3-km spatial block tessellation
- five outer folds
- 500-m fold-specific training exclusion
- final analytical model: Extra Trees with nine locked NISAR-derived predictors
- separate June 25, 2026 temporal-sensitivity experiment

The study is a sensor-sensitivity/methodological evaluation and does not present the final model as an operational wall-to-wall biomass estimator.

## Before Zenodo publication

1. Run a final reproduction check on the analysis machine.
2. Save exact package versions (`python -m pip freeze > requirements_frozen.txt`).
3. Add the exact NISAR and GEDI source-product manifest.
4. Add the final DOI only after the GitHub v1.0.0 release is archived by Zenodo.
