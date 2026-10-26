# Code audit summary

Audited scripts: 28

## Results

- Python syntax: PASS
- Credentials/API keys found: none detected by static text scan
- Original scripts retained unchanged: yes
- Portable copies created: yes
- Main analysis root can be configured with `NISAR_PROJECT_ROOT`
- June workspace can be configured with `NISAR_JUNE_ROOT` / `NISAR_JUNE_PHASE_ROOT`
- Interactive `Press Enter to close` pauses removed only from portable copies
- Stale Phase-6 report text corrected in portable copy: 51 occupied blocks -> 54 occupied blocks
- Documentation-only AOI area in portable Phase-0B copy updated to 595.05 km² for consistency with the final projected analytical AOI

## Final workflow checks visible in the supplied code

- 3-km spatial block size
- five outer spatial folds
- 500-m fold-specific training exclusion
- locked nine-predictor ET9P follow-up
- 5,000 paired block-bootstrap replicates
- held-out permutation importance
- residual Moran's I at 500, 1,000, 2,000 and 3,000 m
- isolated June 25 temporal-sensitivity workflow

## Important limitation

Static code auditing cannot prove numerical reproduction without the original raw/derived input files and the exact Python package versions. Before issuing the Zenodo DOI, run the portable or original workflow on the authors' analysis machine and confirm the final manuscript metrics.
