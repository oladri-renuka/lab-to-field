## 5. File sanity

| dataset                |   files |   unreadable | formats                                                          | modes                                                                               |   EXIF rotated |   min side < 100px | median WxH   |
|:-----------------------|--------:|-------------:|:-----------------------------------------------------------------|:------------------------------------------------------------------------------------|---------------:|-------------------:|:-------------|
| plantdoc               |    2581 |            2 | {'JPEG': np.int64(2571), 'PNG': np.int64(6), 'MPO': np.int64(2)} | {'RGB': np.int64(2567), 'RGBA': np.int64(6), 'CMYK': np.int64(5), 'L': np.int64(1)} |             29 |                  3 | 800x667      |
| plantvillage_color     |   54305 |            0 | {'JPEG': np.int64(54304), 'PNG': np.int64(1)}                    | {'RGB': np.int64(54304), 'RGBA': np.int64(1)}                                       |              0 |                  0 | 256x256      |
| plantvillage_segmented |   28647 |            0 | {'JPEG': np.int64(28647)}                                        | {'RGB': np.int64(28647)}                                                            |              0 |                  0 | 256x256      |

All loading goes through `src/data/io.load_image`: EXIF orientation applied, RGBA/LA/P-with-transparency composited on white, everything else converted to RGB, first frame of GIFs. Unreadable PlantDoc files are dropped in `prepare` (`reports/plantdoc_unreadable.csv`). Tiny images: `plantdoc_tiny.csv`; resolution plot: `plantdoc_resolution.png`.
