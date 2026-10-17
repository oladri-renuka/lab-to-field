# Data audit

## 1a. PlantVillage leaf grouping

- Leaf ids (HF `leaf-map.json`, loader logic) present in **both** train and test: **0** (0 test images). ✅ grouping holds
- Images with no leaf id in the map (fallback, cannot be checked by id): train 24.1%, test 25.0%
- pHash check, PlantVillage test vs train: distance ≤ 4: **260** / 10709 test images; ≤ 10: 5470. Distribution: `pv_test_to_train_phash.png`

## 2. Class counts

| split    |   images |   classes |   min |   max |   imbalance (max/min) |
|:---------|---------:|----------:|------:|------:|----------------------:|
| pv_train |    29622 |        27 |   209 |  4257 |                  20.4 |
| pv_test  |     7244 |        27 |    52 |  1100 |                  21.2 |
| pd_train |     1859 |        27 |    35 |   143 |                   4.1 |
| pd_val   |      465 |        27 |     9 |    36 |                   4   |
| pd_test  |      236 |        27 |     4 |    12 |                   3   |

PlantDoc-test classes with < 10 images: **17** of 27 (Corn_(maize)/Cercospora_leaf_spot Gray_leaf_spot=4, Tomato/Leaf_Mold=6, Tomato/Tomato_Yellow_Leaf_Curl_Virus=6, Squash/Powdery_mildew=6, Raspberry/healthy=7, Potato/Early_blight=8, Pepper,_bell/healthy=8, Potato/Late_blight=8, Strawberry/healthy=8, Tomato/healthy=8, Soybean/healthy=8, Grape/Black_rot=8, Apple/healthy=9, Pepper,_bell/Bacterial_spot=9, Peach/healthy=9, Tomato/Bacterial_spot=9, Tomato/Early_blight=9). Kept; per-class recall for these is reported with that caveat, and overall accuracy carries a bootstrap CI.

## 5. File sanity

| dataset                |   files |   unreadable | formats                                                          | modes                                                                               |   EXIF rotated |   min side < 100px | median WxH   |
|:-----------------------|--------:|-------------:|:-----------------------------------------------------------------|:------------------------------------------------------------------------------------|---------------:|-------------------:|:-------------|
| plantdoc               |    2581 |            2 | {'JPEG': np.int64(2571), 'PNG': np.int64(6), 'MPO': np.int64(2)} | {'RGB': np.int64(2567), 'RGBA': np.int64(6), 'CMYK': np.int64(5), 'L': np.int64(1)} |             29 |                  3 | 800x667      |
| plantvillage_color     |   54305 |            0 | {'JPEG': np.int64(54304), 'PNG': np.int64(1)}                    | {'RGB': np.int64(54304), 'RGBA': np.int64(1)}                                       |              0 |                  0 | 256x256      |
| plantvillage_segmented |   28647 |            0 | {'JPEG': np.int64(28647)}                                        | {'RGB': np.int64(28647)}                                                            |              0 |                  0 | 256x256      |

All loading goes through `src/data/io.load_image`: EXIF orientation applied, RGBA/LA/P-with-transparency composited on white, everything else converted to RGB, first frame of GIFs. Unreadable PlantDoc files are dropped in `prepare` (`reports/plantdoc_unreadable.csv`). Tiny images: `plantdoc_tiny.csv`; resolution plot: `plantdoc_resolution.png`.

## 6. Lab vs field statistics (per image, resized to 256×256; mean ± std across images)

|                    |    n | r             | g             | b             | r_std         | g_std         | b_std         | brightness    | saturation    | blur_lapvar     |
|:-------------------|-----:|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:----------------|
| PlantVillage train | 2000 | 0.469 ± 0.082 | 0.495 ± 0.080 | 0.426 ± 0.087 | 0.183 ± 0.048 | 0.155 ± 0.044 | 0.196 ± 0.042 | 0.480 ± 0.079 | 0.249 ± 0.072 | 63.150 ± 58.344 |
| PlantDoc train+val | 2000 | 0.478 ± 0.166 | 0.545 ± 0.146 | 0.378 ± 0.172 | 0.205 ± 0.052 | 0.197 ± 0.051 | 0.213 ± 0.081 | 0.506 ± 0.148 | 0.414 ± 0.160 | 10.746 ± 8.822  |
| PlantDoc test      |  236 | 0.534 ± 0.200 | 0.608 ± 0.181 | 0.437 ± 0.215 | 0.217 ± 0.056 | 0.198 ± 0.058 | 0.238 ± 0.099 | 0.566 ± 0.186 | 0.391 ± 0.168 | 8.578 ± 8.045   |

Higher Laplacian variance = sharper. Histograms: `domain_stats.png`.

## 7. Segmented config

- Shared-class color-train images with a segmented counterpart (by filename): **30123 / 31098**
- Segmented images with no color counterpart: 1197 of 54306
- Segmented-*train* items whose color version is in color-*test*: **8291** (the two configs were split differently; compositing uses color-train counterparts only, so these never enter training)
- Mask quality on 3000 sampled leaves:

|                                              |   images |
|:---------------------------------------------|---------:|
| size differs from color                      |        0 |
| mask area < 5% (leaf mostly cut away)        |        2 |
| mask area > 90% (background left in)         |       23 |
| > 1 large fragment (specks / split leaf)     |       87 |
| pixel mismatch vs color > 0.10 (not aligned) |      430 |

Median mask area 0.47, median alignment error 0.076. Worst cases (color | segmented): `segmented_worst.jpg`; all rows: `segmented_mask_quality.csv`.

## 4. Label quality

Contact sheets: `reports/audit/labels/{plantdoc,plantvillage}/<class>.jpg` (20 PlantDoc / 10 PlantVillage per class, numbered tiles). Fill the `correct` column of `label_review.csv` with 1/0, then run `python -m src.data.audit --score-labels`.
