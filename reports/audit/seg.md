## 7. Segmented config

- Shared-class color-train images with a segmented counterpart (by filename): **31098 / 31098**
- Segmented images with no color counterpart: 1 of 54306
- Segmented-*train* items whose color version is in color-*test*: **8469** (the two configs were split differently; compositing uses color-train counterparts only, so these never enter training)
- Mask quality on 3000 sampled leaves:

|                                                 |   images |
|:------------------------------------------------|---------:|
| size differs from color                         |        0 |
| mask area < 5% (leaf mostly cut away)           |        0 |
| mask area > 90% (background left in)            |       28 |
| > 1 large fragment (specks / split leaf)        |       94 |
| pixel mismatch vs color > 0.10                  |      416 |
| lesions cut away: repair restores > 10% of leaf |       33 |

Median mask area 0.47, median alignment error 0.076, median repair gain 0.005.

The PlantVillage segmentation removes dark necrotic lesions as if they were background. Compositing therefore uses a repaired mask (closing x7 + hole fill) over the *color* image, and the shortcut test treats the leaf's convex hull as leaf. Largest repairs (color | segmented | pasted): `segmented_repaired.jpg`; worst raw masks: `segmented_worst.jpg`; all rows: `segmented_mask_quality.csv`.
