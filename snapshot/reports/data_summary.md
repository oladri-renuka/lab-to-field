# Data summary

Shared classes (label space): **27**

- PlantDoc images matching a PlantVillage image, removed: **1** (`reports/contamination_plantvillage_in_plantdoc.csv`)
- PlantDoc-train near-duplicates of test, removed: **15** (`reports/dedup_plantdoc.csv`)
- PlantVillage test images near-identical to a train image, removed from test: **246** (`reports/pv_test_near_dup_train.csv`)
- Unreadable PlantDoc files skipped: **0**
- Compositing leaves, color-train only: **31098** (segmented-train items whose color version is in test, excluded: 8469)
- Dropped mapped classes: ['Potato___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite']

| class_name                                         |   pv_train |   pv_test |   pv_seg_train |   pd_train |   pd_val |   pd_test |
|:---------------------------------------------------|-----------:|----------:|---------------:|-----------:|---------:|----------:|
| Apple___Apple_scab                                 |        503 |       127 |            503 |         66 |       17 |        10 |
| Apple___Cedar_apple_rust                           |        223 |        52 |            223 |         63 |       16 |        10 |
| Apple___healthy                                    |       1359 |       282 |           1359 |         66 |       16 |         9 |
| Blueberry___healthy                                |       1178 |       324 |           1178 |         84 |       21 |        11 |
| Cherry_(including_sour)___healthy                  |        731 |       117 |            731 |         38 |        9 |        10 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot |        406 |       107 |            406 |         50 |       12 |         4 |
| Corn_(maize)___Common_rust_                        |        975 |       217 |            975 |         85 |       21 |        10 |
| Corn_(maize)___Northern_Leaf_Blight                |        765 |       219 |            765 |        143 |       36 |        12 |
| Grape___Black_rot                                  |       1012 |       167 |           1012 |         45 |       11 |         8 |
| Grape___healthy                                    |        323 |       100 |            323 |         46 |       11 |        12 |
| Peach___healthy                                    |        289 |        71 |            289 |         82 |       21 |         9 |
| Pepper,_bell___Bacterial_spot                      |        831 |       164 |            831 |         50 |       12 |         9 |
| Pepper,_bell___healthy                             |       1195 |       274 |           1195 |         42 |       11 |         8 |
| Potato___Early_blight                              |        836 |       164 |            836 |         85 |       21 |         8 |
| Potato___Late_blight                               |        788 |       210 |            788 |         77 |       19 |         8 |
| Raspberry___healthy                                |        209 |       142 |            209 |         90 |       22 |         7 |
| Soybean___healthy                                  |       4117 |       804 |           4117 |         46 |       11 |         8 |
| Squash___Powdery_mildew                            |       1457 |       378 |           1457 |         99 |       25 |         6 |
| Strawberry___healthy                               |        376 |        80 |            376 |         70 |       18 |         8 |
| Tomato___Bacterial_spot                            |       1691 |       431 |           1691 |         80 |       20 |         9 |
| Tomato___Early_blight                              |        785 |       214 |            785 |         62 |       16 |         9 |
| Tomato___Late_blight                               |       1534 |       372 |           1534 |         78 |       20 |        10 |
| Tomato___Leaf_Mold                                 |        780 |       172 |            780 |         68 |       17 |         6 |
| Tomato___Septoria_leaf_spot                        |       1434 |       335 |           1434 |        110 |       28 |        11 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus             |       4257 |      1090 |           4257 |         55 |       14 |         6 |
| Tomato___Tomato_mosaic_virus                       |        307 |        65 |            307 |         35 |        9 |        10 |
| Tomato___healthy                                   |       1261 |       324 |           1261 |         44 |       11 |         8 |
