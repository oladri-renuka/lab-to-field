## 1a. PlantVillage leaf grouping

- Leaf ids (HF `leaf-map.json`, loader logic) present in **both** train and test: **0** (0 test images). ✅ grouping holds
- Images with no leaf id in the map (fallback, cannot be checked by id): train 24.1%, test 25.0%
- pHash check, PlantVillage test vs train: distance ≤ 4: **260** / 10709 test images; ≤ 10: 5470. Distribution: `pv_test_to_train_phash.png`
