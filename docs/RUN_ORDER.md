# Recommended execution order

This repository contains two code copies:

- `code/executed_original/`: exact scripts supplied from the authors' analysis workspace. These preserve the executed Windows paths and interactive pauses for provenance.
- `code/portable/`: minimally cleaned copies with environment-variable path configuration and interactive pauses removed. Core analytical logic is otherwise preserved.

## Environment variables

For the primary September workflow:

```powershell
$env:NISAR_PROJECT_ROOT = "D:\NISAR_Mangrove_Paper"
```

For the isolated June temporal-sensitivity workspace:

```powershell
$env:NISAR_JUNE_ROOT = "D:\new_nisar_data"
$env:NISAR_JUNE_PHASE_ROOT = "D:\new_nisar_data\Phase8B2_June25_Sensitivity_V2"
```

## Execution order

1. `00_data_acceptance/check_nisar_gedi_data.py`
2. `00_data_acceptance/create_phase0b_records.py`
3. `01_preprocessing/process_gedi_phase1.py`
4. `01_preprocessing/process_mangrove_mask_phase2.py`
5. `01_preprocessing/process_nisar_phase3.py`
6. `02_matching_eda/match_gedi_nisar_phase4.py`
7. `02_matching_eda/phase5_eda.py`
8. `02_matching_eda/phase5b_spatial_diagnostics.py`
9. `03_spatial_validation/create_phase6_spatial_cv.py`
10. `03_spatial_validation/phase6b_buffered_spatial_cv.py`
11. `04_initial_models/phase7a_baseline_regression.py`
12. `04_initial_models/phase7b_random_forest.py`
13. `04_initial_models/phase7c_xgboost.py`
14. `04_initial_models/phase7d_svr.py`
15. `04_initial_models/phase7e_extra_trees.py`
16. `04_initial_models/phase7f_mlp_dnn.py`
17. `04_initial_models/phase7g_gam.py`
18. `05_sensitivity/phase8a_saturation_analysis.py`
19. `06_multiscale_expansion/phase8c0_september_hdf5_inventory.py`
20. `06_multiscale_expansion/phase8c1_multiscale_predictors.py`
21. `06_multiscale_expansion/phase8c2_match_expanded_predictors.py`
22. `06_multiscale_expansion/phase8c3_predictor_screening.py`
23. `06_multiscale_expansion/phase8c4_expanded9p_spatial_cv.py`
24. `07_final_et9p/phase9_et9p_final_defensibility.py`
25. `07_final_et9p/phase10_final_consolidation.py`

The June 25 experiment is intentionally isolated and should not overwrite the September workflow:

26. `08_temporal_sensitivity/phase8b2a_june25_chunked_V2.py`
27. `08_temporal_sensitivity/phase8b2b_june25_isolated_gedi_matching.py`
28. `08_temporal_sensitivity/phase8b2c_june25_locked_model_sensitivity.py`
