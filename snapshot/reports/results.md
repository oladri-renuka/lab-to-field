# Results (PlantDoc test: 236 images, 27 shared classes)

Headline: **PlantDoc test accuracy** (mean ± std over seeds). The gap is secondary (fine-tuning on field images also lowers lab accuracy, which shrinks the gap without any field gain).

| Exp | Model | **PlantDoc acc** | macro-F1 | ECE | PlantVillage acc | Gap (PV−PD) | PlantDoc acc, PD-val-selected† | selected on | seeds |
|---|---|---|---|---|---|---|---|---|---|
| ddp_ws1 | resnet50 | **0.288 ± 0.000** | 0.272 ± 0.000 | 0.306 ± 0.000 | 0.994 ± 0.000 | 0.705 ± 0.000 | 0.309 ± 0.000 | pv_val | 1 |
| ddp_ws2 | resnet50 | **0.258 ± 0.000** | 0.234 ± 0.000 | 0.298 ± 0.000 | 0.994 ± 0.000 | 0.736 ± 0.000 | 0.271 ± 0.000 | pv_val | 1 |
| ddp_ws4 | resnet50 | **0.237 ± 0.000** | 0.220 ± 0.000 | 0.366 ± 0.000 | 0.994 ± 0.000 | 0.756 ± 0.000 | 0.250 ± 0.000 | pv_val | 1 |

† Optimistic: checkpoint chosen on PlantDoc (field) val. Only reported for rows whose headline checkpoint is chosen on PlantVillage (lab) val.

## Secondary field test: PlantDoc train split (E0/E1 only, which never see PlantDoc pixels)

| Exp | Model | PlantDoc-train acc | seeds |
|---|---|---|---|
| ddp_ws1 | resnet50 | 0.236 ± 0.000 | 1 |
| ddp_ws2 | resnet50 | 0.219 ± 0.000 | 1 |
| ddp_ws4 | resnet50 | 0.208 ± 0.000 | 1 |




## Compute experiments

| run     | model    |   world |   global_batch |   img_per_s |   peak_mem_gb |   epoch_s |   data_wait |   pd_acc |   scaling_eff |
|:--------|:---------|--------:|---------------:|------------:|--------------:|----------:|------------:|---------:|--------------:|
| ddp_ws1 | resnet50 |       1 |            128 |     686.539 |         6.16  |    38.78  |       0.028 |    0.288 |         1     |
| ddp_ws2 | resnet50 |       2 |            256 |    1276.9   |         6.257 |    20.851 |       0.052 |    0.258 |         0.93  |
| ddp_ws4 | resnet50 |       4 |            512 |    2423.48  |         6.257 |    10.986 |       0.095 |    0.237 |         0.882 |
