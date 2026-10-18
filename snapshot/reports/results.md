# Results (PlantDoc test: 236 images, 27 shared classes)

Headline: **PlantDoc test accuracy** (mean ± std over seeds). The gap is secondary (fine-tuning on field images also lowers lab accuracy, which shrinks the gap without any field gain).

| Exp | Model | **PlantDoc acc** | PlantDoc acc, reviewed labels | macro-F1 | ECE | PlantVillage acc | Gap (PV−PD) | PlantDoc acc, PD-val-selected† | selected on | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | resnet50 | **0.291 ± 0.005** | 0.291 ± 0.008 | 0.259 ± 0.011 | 0.283 ± 0.024 | 0.993 ± 0.001 | 0.702 ± 0.005 | 0.295 ± 0.040 | pv_val | 3 |
| e0 | efficientnet_b0 | **0.242 ± 0.004** | 0.243 ± 0.004 | 0.226 ± 0.006 | 0.213 ± 0.032 | 0.995 ± 0.000 | 0.753 ± 0.004 | 0.260 ± 0.026 | pv_val | 3 |
| e0 | vit_s16 | **0.383 ± 0.021** | 0.386 ± 0.021 | 0.365 ± 0.022 | 0.230 ± 0.031 | 0.996 ± 0.000 | 0.614 ± 0.021 | 0.377 ± 0.017 | pv_val | 3 |
| e0 | vit_s16_in1k | **0.264 ± 0.013** | 0.267 ± 0.013 | 0.256 ± 0.022 | 0.321 ± 0.064 | 0.996 ± 0.001 | 0.731 ± 0.013 | 0.295 ± 0.012 | pv_val | 3 |
| e1 | resnet50 | **0.301 ± 0.004** | 0.301 ± 0.005 | 0.273 ± 0.008 | 0.255 ± 0.024 | 0.995 ± 0.001 | 0.694 ± 0.004 | 0.343 ± 0.026 | pv_val | 3 |
| e1 | efficientnet_b0 | **0.244 ± 0.012** | 0.246 ± 0.013 | 0.212 ± 0.012 | 0.224 ± 0.039 | 0.995 ± 0.001 | 0.751 ± 0.012 | 0.267 ± 0.004 | pv_val | 3 |
| e1 | vit_s16 | **0.336 ± 0.016** | 0.339 ± 0.019 | 0.309 ± 0.021 | 0.302 ± 0.047 | 0.997 ± 0.001 | 0.661 ± 0.017 | 0.391 ± 0.018 | pv_val | 3 |
| e1 | vit_s16_in1k | **0.291 ± 0.017** | 0.291 ± 0.016 | 0.262 ± 0.012 | 0.284 ± 0.024 | 0.997 ± 0.001 | 0.706 ± 0.017 | 0.306 ± 0.013 | pv_val | 3 |
| e2 | resnet50 | **0.347 ± 0.017** | 0.357 ± 0.017 | 0.326 ± 0.025 | 0.281 ± 0.012 | 0.992 ± 0.001 | 0.645 ± 0.018 | 0.357 ± 0.027 | pv_val | 3 |
| e2 | efficientnet_b0 | **0.332 ± 0.011** | 0.341 ± 0.011 | 0.303 ± 0.009 | 0.267 ± 0.015 | 0.995 ± 0.001 | 0.663 ± 0.010 | 0.369 ± 0.031 | pv_val | 3 |
| e2 | vit_s16 | **0.366 ± 0.036** | 0.375 ± 0.037 | 0.352 ± 0.025 | 0.303 ± 0.055 | 0.995 ± 0.003 | 0.629 ± 0.036 | 0.468 ± 0.019 | pv_val | 3 |
| e2 | vit_s16_in1k | **0.333 ± 0.038** | 0.342 ± 0.039 | 0.315 ± 0.043 | 0.349 ± 0.024 | 0.996 ± 0.001 | 0.663 ± 0.038 | 0.394 ± 0.018 | pv_val | 3 |
| e3 | resnet50 | **0.326 ± 0.019** | 0.330 ± 0.020 | 0.298 ± 0.010 | 0.329 ± 0.022 | 0.992 ± 0.001 | 0.666 ± 0.019 | 0.340 ± 0.005 | pv_val | 3 |
| e3 | efficientnet_b0 | **0.290 ± 0.029** | 0.291 ± 0.027 | 0.273 ± 0.023 | 0.300 ± 0.036 | 0.994 ± 0.001 | 0.704 ± 0.028 | 0.335 ± 0.019 | pv_val | 3 |
| e3 | vit_s16 | **0.329 ± 0.009** | 0.335 ± 0.008 | 0.310 ± 0.002 | 0.385 ± 0.028 | 0.995 ± 0.002 | 0.666 ± 0.008 | 0.455 ± 0.018 | pv_val | 3 |
| e4 | resnet50 | **0.302 ± 0.034** | 0.304 ± 0.036 | 0.279 ± 0.035 | 0.234 ± 0.036 | 0.992 ± 0.001 | 0.690 ± 0.035 | 0.331 ± 0.046 | pv_val | 3 |
| e4 | efficientnet_b0 | **0.261 ± 0.009** | 0.262 ± 0.011 | 0.246 ± 0.017 | 0.153 ± 0.019 | 0.995 ± 0.001 | 0.734 ± 0.008 | 0.268 ± 0.005 | pv_val | 3 |
| e4 | vit_s16 | **0.394 ± 0.022** | 0.394 ± 0.021 | 0.362 ± 0.035 | 0.126 ± 0.064 | 0.992 ± 0.003 | 0.598 ± 0.023 | 0.442 ± 0.018 | pv_val | 3 |
| e5_k5 | resnet50 | **0.386 ± 0.043** | 0.388 ± 0.042 | 0.377 ± 0.031 | 0.100 ± 0.037 | 0.932 ± 0.011 | 0.547 ± 0.039 | — | pd_val | 3 |
| e5_k5 | efficientnet_b0 | **0.431 ± 0.034** | 0.435 ± 0.036 | 0.421 ± 0.027 | 0.101 ± 0.011 | 0.942 ± 0.010 | 0.511 ± 0.043 | — | pd_val | 3 |
| e5_k5 | vit_s16 | **0.559 ± 0.034** | 0.567 ± 0.032 | 0.540 ± 0.037 | 0.110 ± 0.028 | 0.992 ± 0.004 | 0.433 ± 0.035 | — | pd_val | 3 |
| e5_k10 | resnet50 | **0.446 ± 0.016** | 0.449 ± 0.016 | 0.439 ± 0.021 | 0.101 ± 0.022 | 0.947 ± 0.002 | 0.501 ± 0.017 | — | pd_val | 3 |
| e5_k10 | efficientnet_b0 | **0.476 ± 0.028** | 0.481 ± 0.031 | 0.470 ± 0.025 | 0.096 ± 0.005 | 0.941 ± 0.006 | 0.465 ± 0.027 | — | pd_val | 3 |
| e5_k10 | vit_s16 | **0.592 ± 0.032** | 0.597 ± 0.031 | 0.574 ± 0.035 | 0.109 ± 0.032 | 0.993 ± 0.003 | 0.401 ± 0.033 | — | pd_val | 3 |
| e5_k25 | resnet50 | **0.517 ± 0.034** | 0.525 ± 0.032 | 0.511 ± 0.046 | 0.116 ± 0.027 | 0.944 ± 0.005 | 0.427 ± 0.036 | — | pd_val | 3 |
| e5_k25 | efficientnet_b0 | **0.558 ± 0.018** | 0.562 ± 0.020 | 0.553 ± 0.016 | 0.081 ± 0.021 | 0.938 ± 0.015 | 0.380 ± 0.027 | — | pd_val | 3 |
| e5_k25 | vit_s16 | **0.650 ± 0.030** | 0.655 ± 0.026 | 0.643 ± 0.029 | 0.095 ± 0.006 | 0.991 ± 0.005 | 0.341 ± 0.035 | — | pd_val | 3 |
| e6_from_e2 | resnet50 | **0.477 ± 0.014** | 0.484 ± 0.016 | 0.460 ± 0.015 | 0.139 ± 0.015 | 0.943 ± 0.015 | 0.465 ± 0.024 | — | pd_val | 3 |
| e6_from_e2 | efficientnet_b0 | **0.518 ± 0.038** | 0.529 ± 0.036 | 0.499 ± 0.039 | 0.161 ± 0.026 | 0.935 ± 0.068 | 0.417 ± 0.091 | — | pd_val | 3 |
| e6_from_e2 | vit_s16 | **0.559 ± 0.028** | 0.568 ± 0.027 | 0.542 ± 0.028 | 0.148 ± 0.012 | 0.991 ± 0.005 | 0.432 ± 0.030 | — | pd_val | 3 |
| e7 | resnet50 | **0.638 ± 0.009** | 0.646 ± 0.005 | 0.627 ± 0.004 | 0.075 ± 0.011 | 0.384 ± 0.026 | -0.255 ± 0.030 | — | pd_val | 3 |
| e7 | efficientnet_b0 | **0.541 ± 0.009** | 0.543 ± 0.008 | 0.526 ± 0.014 | 0.075 ± 0.037 | 0.305 ± 0.037 | -0.236 ± 0.040 | — | pd_val | 3 |
| e7 | vit_s16 | **0.725 ± 0.019** | 0.729 ± 0.018 | 0.701 ± 0.023 | 0.091 ± 0.010 | 0.536 ± 0.043 | -0.189 ± 0.053 | — | pd_val | 3 |

† Optimistic: checkpoint chosen on PlantDoc (field) val. Only reported for rows whose headline checkpoint is chosen on PlantVillage (lab) val.

## Secondary field test: PlantDoc train split (E0/E1 only, which never see PlantDoc pixels)

| Exp | Model | PlantDoc-train acc | seeds |
|---|---|---|---|
| e0 | resnet50 | 0.238 ± 0.008 | 3 |
| e0 | efficientnet_b0 | 0.174 ± 0.015 | 3 |
| e0 | vit_s16 | 0.303 ± 0.029 | 3 |
| e0 | vit_s16_in1k | 0.201 ± 0.021 | 3 |
| e1 | resnet50 | 0.230 ± 0.003 | 3 |
| e1 | efficientnet_b0 | 0.192 ± 0.010 | 3 |
| e1 | vit_s16 | 0.283 ± 0.015 | 3 |
| e1 | vit_s16_in1k | 0.241 ± 0.011 | 3 |

## Share of the achievable field gain recovered: (E − E0) / (E7 − E0)

E0 = lab only, E7 = PlantDoc train only (reference). Computed on mean PlantDoc accuracy; significance is in the paired-comparison table.

| Exp | resnet50 | efficientnet_b0 | vit_s16 | vit_s16_in1k |
|---|---|---|---|---|
| e1 | +3% | +1% | -14% | +nan% |
| e2 | +16% | +30% | -5% | +nan% |
| e3 | +10% | +16% | -16% | — |
| e4 | +3% | +7% | +3% | — |
| e5_k5 | +27% | +63% | +52% | — |
| e5_k10 | +45% | +78% | +61% | — |
| e5_k25 | +65% | +106% | +78% | — |
| e6 | +54% | +92% | +52% | — |

## Paired comparisons on the same PlantDoc test images

Per-image correctness averaged over each cell's seeds; paired bootstrap over images (10k resamples) is the primary test, exact McNemar on majority-vote correctness is supplementary. **One Holm family** covers all 72 comparisons (within-model experiment pairs and across-model pairs). `<1e-4` = no bootstrap resample crossed zero. With 236 test images, differences under ~5 points are unlikely to be detectable.

Across-model rows: `vit_s16` (ImageNet-21k) vs `vit_s16_in1k` (same model and recipe, ImageNet-1k only) measures the pretraining-data effect; `vit_s16_in1k` vs the CNNs (also ImageNet-1k) measures architecture at matched pretraining data.

| kind          | model / exp     | A            | B               |   ΔPlantDoc acc (A−B) | 95% CI           | p (paired bootstrap)   | p McNemar   | seeds A/B   | p Holm   | p McNemar Holm   | significant (Holm < 0.05)   |
|:--------------|:----------------|:-------------|:----------------|----------------------:|:-----------------|:-----------------------|:------------|:------------|:---------|:-----------------|:----------------------------|
| within model  | resnet50        | e1           | e0              |                0.0099 | [-0.028, +0.049] | 0.6144                 | 0.6271      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | resnet50        | e2           | e0              |                0.0565 | [+0.011, +0.103] | 0.0150                 | 0.1742      | 3/3         | 0.4950   | 1.0000           | no                          |
| within model  | resnet50        | e3           | e0              |                0.0353 | [-0.007, +0.078] | 0.1032                 | 0.1996      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | resnet50        | e4           | e0              |                0.0113 | [-0.021, +0.044] | 0.4926                 | 1.0000      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | resnet50        | e5_k10       | e0              |                0.1554 | [+0.105, +0.206] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | resnet50        | e5_k25       | e0              |                0.226  | [+0.169, +0.284] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | resnet50        | e5_k5        | e0              |                0.0946 | [+0.048, +0.143] | <1e-4                  | 0.0018      | 3/3         | <1e-4    | 0.0773           | yes                         |
| within model  | resnet50        | e6           | e0              |                0.1864 | [+0.130, +0.243] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | resnet50        | e7           | e0              |                0.3475 | [+0.285, +0.410] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | resnet50        | e2           | e1              |                0.0466 | [-0.000, +0.093] | 0.0538                 | 0.4709      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | resnet50        | e3           | e2              |               -0.0212 | [-0.076, +0.032] | 0.4586                 | 1.0000      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | resnet50        | e4           | e2              |               -0.0452 | [-0.092, +0.001] | 0.0558                 | 0.1839      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | resnet50        | e5_k10       | e3              |                0.1201 | [+0.073, +0.168] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | 0.0016           | yes                         |
| within model  | resnet50        | e5_k10       | e4              |                0.1441 | [+0.092, +0.199] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | 0.0001           | yes                         |
| within model  | resnet50        | e5_k10       | e2              |                0.0989 | [+0.045, +0.154] | <1e-4                  | 0.0004      | 3/3         | <1e-4    | 0.0198           | yes                         |
| within model  | resnet50        | e6           | e5_k10          |                0.0311 | [-0.014, +0.075] | 0.1790                 | 0.1641      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | resnet50        | e6           | e2              |                0.1299 | [+0.079, +0.182] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | efficientnet_b0 | e1           | e0              |                0.0028 | [-0.031, +0.038] | 0.8766                 | 0.6177      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | efficientnet_b0 | e2           | e0              |                0.0904 | [+0.045, +0.137] | <1e-4                  | 0.0007      | 3/3         | <1e-4    | 0.0316           | yes                         |
| within model  | efficientnet_b0 | e3           | e0              |                0.048  | [+0.006, +0.092] | 0.0270                 | 0.0186      | 3/3         | 0.8100   | 0.7077           | no                          |
| within model  | efficientnet_b0 | e4           | e0              |                0.0198 | [-0.013, +0.052] | 0.2454                 | 0.6076      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | efficientnet_b0 | e5_k10       | e0              |                0.2345 | [+0.182, +0.288] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | efficientnet_b0 | e5_k25       | e0              |                0.3164 | [+0.260, +0.373] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | efficientnet_b0 | e5_k5        | e0              |                0.1893 | [+0.140, +0.240] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | efficientnet_b0 | e6           | e0              |                0.2768 | [+0.222, +0.333] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | efficientnet_b0 | e7           | e0              |                0.2994 | [+0.237, +0.360] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | efficientnet_b0 | e2           | e1              |                0.0876 | [+0.038, +0.136] | 0.0012                 | 0.0078      | 3/3         | 0.0480   | 0.3115           | yes                         |
| within model  | efficientnet_b0 | e3           | e2              |               -0.0424 | [-0.089, +0.007] | 0.0856                 | 0.4101      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | efficientnet_b0 | e4           | e2              |               -0.0706 | [-0.120, -0.020] | 0.0046                 | 0.0078      | 3/3         | 0.1702   | 0.3115           | no                          |
| within model  | efficientnet_b0 | e5_k10       | e3              |                0.1864 | [+0.131, +0.240] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | 0.0002           | yes                         |
| within model  | efficientnet_b0 | e5_k10       | e4              |                0.2147 | [+0.165, +0.266] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | efficientnet_b0 | e5_k10       | e2              |                0.1441 | [+0.085, +0.203] | <1e-4                  | 0.0002      | 3/3         | <1e-4    | 0.0084           | yes                         |
| within model  | efficientnet_b0 | e6           | e5_k10          |                0.0424 | [-0.003, +0.088] | 0.0658                 | 0.0489      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | efficientnet_b0 | e6           | e2              |                0.1864 | [+0.138, +0.236] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16         | e1           | e0              |               -0.0466 | [-0.078, -0.016] | 0.0044                 | 0.1078      | 3/3         | 0.1672   | 1.0000           | no                          |
| within model  | vit_s16         | e2           | e0              |               -0.0169 | [-0.064, +0.030] | 0.4706                 | 0.6655      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | vit_s16         | e3           | e0              |               -0.0537 | [-0.095, -0.011] | 0.0152                 | 0.2221      | 3/3         | 0.4950   | 1.0000           | no                          |
| within model  | vit_s16         | e4           | e0              |                0.0113 | [-0.023, +0.045] | 0.5290                 | 0.8679      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | vit_s16         | e5_k10       | e0              |                0.209  | [+0.164, +0.256] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16         | e5_k25       | e0              |                0.2669 | [+0.213, +0.321] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16         | e5_k5        | e0              |                0.1766 | [+0.134, +0.220] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16         | e6           | e0              |                0.1766 | [+0.123, +0.232] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16         | e7           | e0              |                0.3418 | [+0.285, +0.397] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16         | e2           | e1              |                0.0297 | [-0.018, +0.078] | 0.2338                 | 0.5682      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | vit_s16         | e3           | e2              |               -0.0367 | [-0.088, +0.014] | 0.1708                 | 0.6147      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | vit_s16         | e4           | e2              |                0.0282 | [-0.018, +0.075] | 0.2350                 | 0.4966      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | vit_s16         | e5_k10       | e3              |                0.2627 | [+0.212, +0.314] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16         | e5_k10       | e4              |                0.1977 | [+0.151, +0.247] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16         | e5_k10       | e2              |                0.226  | [+0.178, +0.274] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16         | e6           | e5_k10          |               -0.0325 | [-0.072, +0.006] | 0.0946                 | 0.7493      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | vit_s16         | e6           | e2              |                0.1935 | [+0.151, +0.237] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| within model  | vit_s16_in1k    | e1           | e0              |                0.0268 | [-0.008, +0.062] | 0.1372                 | 0.1214      | 3/3         | 1.0000   | 1.0000           | no                          |
| within model  | vit_s16_in1k    | e2           | e0              |                0.0692 | [+0.027, +0.113] | 0.0016                 | 0.0725      | 3/3         | 0.0624   | 1.0000           | no                          |
| within model  | vit_s16_in1k    | e2           | e1              |                0.0424 | [-0.004, +0.088] | 0.0702                 | 0.7709      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e0              | vit_s16      | resnet50        |                0.0918 | [+0.049, +0.134] | <1e-4                  | 0.0045      | 3/3         | <1e-4    | 0.1859           | yes                         |
| across models | e0              | vit_s16      | efficientnet_b0 |                0.1412 | [+0.099, +0.185] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | <1e-4            | yes                         |
| across models | e0              | resnet50     | efficientnet_b0 |                0.0494 | [+0.010, +0.089] | 0.0144                 | 0.0237      | 3/3         | 0.4896   | 0.8770           | no                          |
| across models | e0              | vit_s16_in1k | resnet50        |               -0.0268 | [-0.071, +0.017] | 0.2162                 | 0.3368      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e0              | vit_s16_in1k | efficientnet_b0 |                0.0226 | [-0.013, +0.059] | 0.2210                 | 0.2153      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e0              | vit_s16      | vit_s16_in1k    |                0.1186 | [+0.081, +0.157] | <1e-4                  | <1e-4       | 3/3         | <1e-4    | 0.0012           | yes                         |
| across models | e1              | vit_s16      | resnet50        |                0.0353 | [+0.000, +0.071] | 0.0474                 | 0.3105      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e1              | vit_s16      | efficientnet_b0 |                0.0918 | [+0.051, +0.133] | <1e-4                  | 0.0009      | 3/3         | <1e-4    | 0.0404           | yes                         |
| across models | e1              | resnet50     | efficientnet_b0 |                0.0565 | [+0.014, +0.100] | 0.0080                 | 0.0315      | 3/3         | 0.2880   | 1.0000           | no                          |
| across models | e1              | vit_s16_in1k | resnet50        |               -0.0099 | [-0.052, +0.031] | 0.6220                 | 1.0000      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e1              | vit_s16_in1k | efficientnet_b0 |                0.0466 | [+0.006, +0.086] | 0.0244                 | 0.0436      | 3/3         | 0.7564   | 1.0000           | no                          |
| across models | e1              | vit_s16      | vit_s16_in1k    |                0.0452 | [+0.010, +0.081] | 0.0108                 | 0.2005      | 3/3         | 0.3780   | 1.0000           | no                          |
| across models | e2              | vit_s16      | resnet50        |                0.0184 | [-0.024, +0.059] | 0.4020                 | 0.4614      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e2              | vit_s16      | efficientnet_b0 |                0.0339 | [-0.008, +0.078] | 0.1296                 | 0.3713      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e2              | resnet50     | efficientnet_b0 |                0.0155 | [-0.020, +0.052] | 0.3944                 | 1.0000      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e2              | vit_s16_in1k | resnet50        |               -0.0141 | [-0.056, +0.028] | 0.5102                 | 0.6440      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e2              | vit_s16_in1k | efficientnet_b0 |                0.0014 | [-0.041, +0.044] | 0.9674                 | 0.7552      | 3/3         | 1.0000   | 1.0000           | no                          |
| across models | e2              | vit_s16      | vit_s16_in1k    |                0.0325 | [-0.008, +0.073] | 0.1166                 | 0.1641      | 3/3         | 1.0000   | 1.0000           | no                          |

## Background-only shortcut test, per architecture (PlantVillage test, leaf-free patches)

Majority-class baseline 0.156; uniform chance 0.037.

| Model | Background-only acc | seeds |
|---|---|---|
| resnet50 | 0.818 ± 0.004 | 3 |
| efficientnet_b0 | 0.859 ± 0.002 | 3 |
| vit_s16 | 0.889 ± 0.002 | 3 |
| vit_s16_in1k | 0.891 ± 0.002 | 3 |

This measures how well each architecture *can* exploit background statistics. How much a lab-trained model *relies* on them is read from the E2−E0 gain and the saliency figures.


