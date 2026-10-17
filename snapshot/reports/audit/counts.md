## 2. Class counts

| split    |   images |   classes |   min |   max |   imbalance (max/min) |
|:---------|---------:|----------:|------:|------:|----------------------:|
| pv_train |    29622 |        27 |   209 |  4257 |                  20.4 |
| pv_test  |     7002 |        27 |    52 |  1090 |                  21   |
| pd_train |     1859 |        27 |    35 |   143 |                   4.1 |
| pd_val   |      465 |        27 |     9 |    36 |                   4   |
| pd_test  |      236 |        27 |     4 |    12 |                   3   |

PlantDoc-test classes with < 10 images: **17** of 27 (Corn_(maize)/Cercospora_leaf_spot Gray_leaf_spot=4, Tomato/Leaf_Mold=6, Tomato/Tomato_Yellow_Leaf_Curl_Virus=6, Squash/Powdery_mildew=6, Raspberry/healthy=7, Potato/Early_blight=8, Pepper,_bell/healthy=8, Potato/Late_blight=8, Strawberry/healthy=8, Tomato/healthy=8, Soybean/healthy=8, Grape/Black_rot=8, Apple/healthy=9, Pepper,_bell/Bacterial_spot=9, Peach/healthy=9, Tomato/Bacterial_spot=9, Tomato/Early_blight=9). Kept; per-class recall for these is reported with that caveat, and overall accuracy carries a bootstrap CI.
