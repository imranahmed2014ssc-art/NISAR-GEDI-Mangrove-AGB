# Reviewer-validation reproducibility package

This directory contains the frozen analyses added during major revision of the NISAR–GEDI mangrove AGBD manuscript.

## Validation modules

- `C1_Independent_Validation` — selection-aware nested geographic validation
- `C2_Spatial_Transferability` — buffer, pass-holdout, and randomized block sensitivity
- `C3_Temporal_Mismatch` — temporal-gap, June-scene, hydrometeorological, and stability analyses
- `C4_GEDI_Uncertainty` — GEDI `agbd_se` uncertainty characterization
- `C5_Multiscale_Ablation` — predeclared multiscale predictor ablation
- `C6_Support_Mixing` — mangrove-support and GCOV-quality sensitivity
- `C8_Reproducibility` — reproducibility audit, environment snapshot, and checksums

## Frozen analytical principles

No C1–C6 predictor family, spatial split, support threshold, or uncertainty stratum was redefined after viewing the corresponding performance results.

C1–C6 are within-study validation analyses and do not constitute independent external field validation.

The manuscript does not present an operational wall-to-wall biomass product.

## Main frozen modeling table

Final matched sample: 1,641 GEDI–NISAR observations.

Frozen master-table SHA256:

`de9fc736ec77d38dae527020b2d037c4567f3c814f874d9843da230c6bf67dc0`

## GitHub release

This reviewer-validation package is intended for release as **v1.0.2**, superseding v1.0.1 for the revised manuscript reproducibility record.

See `docs/SOURCE_PRODUCTS.md` for source-product identifiers.
