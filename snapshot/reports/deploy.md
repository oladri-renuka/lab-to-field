Deployed checkpoint: `runs/e7/efficientnet_b0/seed2/best.pt` (best EfficientNet-B0 by PlantDoc-val accuracy).
The demo serves **model_fp32.onnx** (INT8 is used only if it loses at most 3% PlantDoc-val accuracy; it lost 16.3%).

| Model | PlantDoc-val acc | PlantDoc-test acc | Size (MB) | CPU p50 (ms) | CPU p95 (ms) |
|---|---|---|---|---|---|
| EfficientNet-B0 FP32 | 0.611 | 0.542 | 15.4 | 4.7 | 4.9 |
| EfficientNet-B0 INT8 (percentile_conv) | 0.447 | 0.407 | 4.6 | 9.3 | 11.2 |

CPU: x86_64, 4 threads, batch 1, onnxruntime 1.31.0.

INT8 variants tried (chosen on PlantDoc-val; test not used for the choice):

| Variant | Calibration | Calib. images | Quantized ops | PlantDoc-val acc | Size (MB) | p50 (ms) |
|---|---|---|---|---|---|---|
| minmax_all | MinMax | 500 | all | 0.159 | 4.6 | 7.4 |
| percentile_all | Percentile | 128 | all | 0.424 | 4.6 | 6.6 |
| entropy_all | Entropy | 128 | all | 0.234 | 4.6 | 6.1 |
| minmax_conv | MinMax | 500 | Conv/Gemm/MatMul | 0.185 | 4.6 | 7.9 |
| percentile_conv | Percentile | 128 | Conv/Gemm/MatMul | 0.447 | 4.6 | 9.5 |
| entropy_conv | Entropy | 128 | Conv/Gemm/MatMul | 0.213 | 4.6 | 10.8 |
