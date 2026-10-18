# Data summary

Shared classes (label space): **27**

- PlantDoc images matching a PlantVillage image, removed: **1** (`reports/contamination_plantvillage_in_plantdoc.csv`)
- PlantDoc-train near-duplicates of test, removed: **15** (`reports/dedup_plantdoc.csv`)
- PlantVillage test images near-identical to a train image, removed from test: **246** (`reports/pv_test_near_dup_train.csv`)
- Unreadable PlantDoc files skipped: **0**
- Compositing leaves, color-train only: **27988** (segmented-train items whose color version is in test, excluded: 8469)
- Dropped mapped classes: ['Potato___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite']

| class_name                                         |   pv_train |   pv_val |   pv_test |   pv_seg_train |   pd_train |   pd_val |   pd_test |
|:---------------------------------------------------|-----------:|---------:|----------:|---------------:|-----------:|---------:|----------:|
| Apple___Apple_scab                                 |        454 |       49 |       127 |            454 |         66 |       17 |        10 |
| Apple___Cedar_apple_rust                           |        199 |       24 |        52 |            199 |         63 |       16 |        10 |
| Apple___healthy                                    |       1224 |      134 |       282 |           1224 |         66 |       16 |         9 |
| Blueberry___healthy                                |       1063 |      115 |       324 |           1063 |         84 |       21 |        11 |
| Cherry_(including_sour)___healthy                  |        658 |       73 |       117 |            658 |         38 |        9 |        10 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot |        365 |       41 |       107 |            365 |         50 |       12 |         4 |
| Corn_(maize)___Common_rust_                        |        878 |       97 |       217 |            878 |         85 |       21 |        10 |
| Corn_(maize)___Northern_Leaf_Blight                |        688 |       77 |       219 |            688 |        143 |       36 |        12 |
| Grape___Black_rot                                  |        912 |      100 |       167 |            912 |         45 |       11 |         8 |
| Grape___healthy                                    |        291 |       32 |       100 |            291 |         46 |       11 |        12 |
| Peach___healthy                                    |        261 |       28 |        71 |            261 |         82 |       21 |         9 |
| Pepper,_bell___Bacterial_spot                      |        747 |       84 |       164 |            747 |         50 |       12 |         9 |
| Pepper,_bell___healthy                             |       1076 |      117 |       274 |           1076 |         42 |       11 |         8 |
| Potato___Early_blight                              |        752 |       84 |       164 |            752 |         85 |       21 |         8 |
| Potato___Late_blight                               |        708 |       80 |       210 |            708 |         77 |       19 |         8 |
| Raspberry___healthy                                |        185 |       19 |       142 |            185 |         90 |       22 |         7 |
| Soybean___healthy                                  |       3706 |      339 |       804 |           3706 |         46 |       11 |         8 |
| Squash___Powdery_mildew                            |       1312 |      144 |       378 |           1312 |         99 |       25 |         6 |
| Strawberry___healthy                               |        338 |       38 |        80 |            338 |         70 |       18 |         8 |
| Tomato___Bacterial_spot                            |       1519 |      172 |       431 |           1519 |         80 |       20 |         9 |
| Tomato___Early_blight                              |        705 |       80 |       214 |            705 |         62 |       16 |         9 |
| Tomato___Late_blight                               |       1381 |      153 |       372 |           1381 |         78 |       20 |        10 |
| Tomato___Leaf_Mold                                 |        704 |       76 |       172 |            704 |         68 |       17 |         6 |
| Tomato___Septoria_leaf_spot                        |       1291 |      142 |       335 |           1291 |        110 |       28 |        11 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus             |       3831 |      421 |      1090 |           3831 |         55 |       14 |         6 |
| Tomato___Tomato_mosaic_virus                       |        277 |       30 |        65 |            277 |         35 |        9 |        10 |
| Tomato___healthy                                   |       1135 |      126 |       324 |           1135 |         44 |       11 |         8 |
