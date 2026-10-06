| method             |   block_length |   f1_tau10 |   precision |   recall |   n_predicted_cps |   median_abs_error |
|:-------------------|---------------:|-----------:|------------:|---------:|------------------:|-------------------:|
| block_pointwise    |              1 |      0.665 |       0.651 |    0.709 |            48.8   |              9.492 |
| block_pointwise    |              5 |      0.408 |       0.412 |    0.409 |            36.383 |             11.694 |
| block_pointwise    |             10 |      0.339 |       0.346 |    0.333 |            26.217 |             12.667 |
| block_pointwise    |             25 |      0.321 |       0.329 |    0.314 |            22.6   |             12.05  |
| block_pointwise    |             50 |      0.504 |       0.507 |    0.508 |            43.367 |             11.14  |
| fixed_quantile     |            nan |      0.328 |       0.334 |    0.323 |            48.233 |             16.562 |
| resample_pointwise |              1 |      0.664 |       0.65  |    0.71  |            48.917 |              9.392 |