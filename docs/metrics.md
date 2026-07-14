# ChipTherm Metric Definitions

## RMSE

ChipTherm reports two RMSE aggregations when evaluating full 64x64 temperature maps.

- `global_pixel_rmse_K`: square root of the mean squared error over every sample and every grid cell. This is the primary paper-facing RMSE.
- `mean_sample_rmse_K`: compute one RMSE per sample over its grid cells, then average those sample RMSE values.

For backward compatibility, `rmse_K` is retained as an alias for `global_pixel_rmse_K` in newly generated metrics outputs.

These definitions differ when some samples have larger errors than others. For two 2x2 samples with per-sample RMSE values 0 K and 4 K:

- `global_pixel_rmse_K = sqrt((0^2 + 4^2) / 2) = 2.828 K`
- `mean_sample_rmse_K = (0 + 4) / 2 = 2.000 K`

Historical analysis CSVs that were produced before this standardization may have used `rmse_K` as a mean per-sample RMSE in aggregated summaries. Regenerate those reports to obtain explicit `global_pixel_rmse_K` and `mean_sample_rmse_K` fields.
