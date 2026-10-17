# Results (PlantDoc test: 236 images, 27 shared classes)

Headline: **PlantDoc test accuracy** (mean ± std over seeds). The gap is secondary (fine-tuning on field images also lowers lab accuracy, which shrinks the gap without any field gain).

| Exp | Model | **PlantDoc acc** | macro-F1 | ECE | PlantVillage acc | Gap (PV−PD) | PlantDoc acc, PD-val-selected† | selected on | seeds |
|---|---|---|---|---|---|---|---|---|---|
| e0 | resnet50 | **0.291 ± 0.005** | 0.259 ± 0.011 | 0.283 ± 0.024 | 0.993 ± 0.001 | 0.702 ± 0.005 | 0.295 ± 0.040 | pv_val | 3 |
| e0 | efficientnet_b0 | **0.242 ± 0.004** | 0.226 ± 0.006 | 0.213 ± 0.032 | 0.995 ± 0.000 | 0.753 ± 0.004 | 0.260 ± 0.026 | pv_val | 3 |
| e0 | vit_s16 | **0.383 ± 0.021** | 0.365 ± 0.022 | 0.230 ± 0.031 | 0.996 ± 0.000 | 0.614 ± 0.021 | 0.377 ± 0.017 | pv_val | 3 |
| e1 | resnet50 | **0.301 ± 0.004** | 0.273 ± 0.008 | 0.255 ± 0.024 | 0.995 ± 0.001 | 0.694 ± 0.004 | 0.343 ± 0.026 | pv_val | 3 |
| e1 | efficientnet_b0 | **0.244 ± 0.012** | 0.212 ± 0.012 | 0.224 ± 0.039 | 0.995 ± 0.001 | 0.751 ± 0.012 | 0.267 ± 0.004 | pv_val | 3 |
| e1 | vit_s16 | **0.336 ± 0.016** | 0.309 ± 0.021 | 0.302 ± 0.047 | 0.997 ± 0.001 | 0.661 ± 0.017 | 0.391 ± 0.018 | pv_val | 3 |
| e2 | resnet50 | **0.347 ± 0.017** | 0.326 ± 0.025 | 0.281 ± 0.012 | 0.992 ± 0.001 | 0.645 ± 0.018 | 0.357 ± 0.027 | pv_val | 3 |
| e2 | efficientnet_b0 | **0.332 ± 0.011** | 0.303 ± 0.009 | 0.267 ± 0.015 | 0.995 ± 0.001 | 0.663 ± 0.010 | 0.369 ± 0.031 | pv_val | 3 |
| e2 | vit_s16 | **0.366 ± 0.036** | 0.352 ± 0.025 | 0.303 ± 0.055 | 0.995 ± 0.003 | 0.629 ± 0.036 | 0.468 ± 0.019 | pv_val | 3 |
| e7 | resnet50 | **0.638 ± 0.009** | 0.627 ± 0.004 | 0.075 ± 0.011 | 0.384 ± 0.026 | -0.255 ± 0.030 | — | pd_val | 3 |
| e7 | efficientnet_b0 | **0.541 ± 0.009** | 0.526 ± 0.014 | 0.075 ± 0.037 | 0.305 ± 0.037 | -0.236 ± 0.040 | — | pd_val | 3 |
| e7 | vit_s16 | **0.725 ± 0.019** | 0.701 ± 0.023 | 0.091 ± 0.010 | 0.536 ± 0.043 | -0.189 ± 0.053 | — | pd_val | 3 |

† Optimistic: checkpoint chosen on PlantDoc (field) val. Only reported for rows whose headline checkpoint is chosen on PlantVillage (lab) val.

## Secondary field test: PlantDoc train split (E0/E1 only, which never see PlantDoc pixels)

| Exp | Model | PlantDoc-train acc | seeds |
|---|---|---|---|
| e0 | resnet50 | 0.238 ± 0.008 | 3 |
| e0 | efficientnet_b0 | 0.174 ± 0.015 | 3 |
| e0 | vit_s16 | 0.303 ± 0.029 | 3 |
| e1 | resnet50 | 0.230 ± 0.003 | 3 |
| e1 | efficientnet_b0 | 0.192 ± 0.010 | 3 |
| e1 | vit_s16 | 0.283 ± 0.015 | 3 |

## Paired comparisons on the same PlantDoc test images

Per-image correctness averaged over each experiment's seeds; paired bootstrap over images (10k resamples) is the primary test, exact McNemar on majority-vote correctness is supplementary. Holm correction across all 12 comparisons. With ~236 test images, differences under ~5 points are unlikely to be detectable.

| model           | A   | B   |   ΔPlantDoc acc (A−B) | 95% CI           |   p (paired bootstrap) |   p McNemar | seeds A/B   |   p Holm |   p McNemar Holm | significant (Holm < 0.05)   |
|:----------------|:----|:----|----------------------:|:-----------------|-----------------------:|------------:|:------------|---------:|-----------------:|:----------------------------|
| resnet50        | e1  | e0  |                0.0099 | [-0.028, +0.049] |                 0.6144 |      0.6271 | 3/3         |   1      |           1      | no                          |
| resnet50        | e2  | e0  |                0.0565 | [+0.011, +0.103] |                 0.015  |      0.1742 | 3/3         |   0.09   |           1      | no                          |
| resnet50        | e7  | e0  |                0.3475 | [+0.285, +0.410] |                 0      |      0      | 3/3         |   0      |           0      | yes                         |
| resnet50        | e2  | e1  |                0.0466 | [-0.000, +0.093] |                 0.0538 |      0.4709 | 3/3         |   0.269  |           1      | no                          |
| efficientnet_b0 | e1  | e0  |                0.0028 | [-0.031, +0.038] |                 0.8766 |      0.6177 | 3/3         |   1      |           1      | no                          |
| efficientnet_b0 | e2  | e0  |                0.0904 | [+0.045, +0.137] |                 0      |      0.0007 | 3/3         |   0      |           0.0065 | yes                         |
| efficientnet_b0 | e7  | e0  |                0.2994 | [+0.237, +0.360] |                 0      |      0      | 3/3         |   0      |           0      | yes                         |
| efficientnet_b0 | e2  | e1  |                0.0876 | [+0.038, +0.136] |                 0.0012 |      0.0078 | 3/3         |   0.0096 |           0.0623 | yes                         |
| vit_s16         | e1  | e0  |               -0.0466 | [-0.078, -0.016] |                 0.0044 |      0.1078 | 3/3         |   0.0308 |           0.7543 | yes                         |
| vit_s16         | e2  | e0  |               -0.0169 | [-0.064, +0.030] |                 0.4706 |      0.6655 | 3/3         |   1      |           1      | no                          |
| vit_s16         | e7  | e0  |                0.3418 | [+0.285, +0.397] |                 0      |      0      | 3/3         |   0      |           0      | yes                         |
| vit_s16         | e2  | e1  |                0.0297 | [-0.018, +0.078] |                 0.2338 |      0.5682 | 3/3         |   0.9352 |           1      | no                          |

