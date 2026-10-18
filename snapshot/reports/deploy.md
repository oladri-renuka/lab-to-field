Deployed checkpoint: `runs/e7/efficientnet_b0/seed2/best.pt` (best EfficientNet-B0 by PlantDoc-val accuracy).
The demo serves **model_fp32.onnx** (INT8 is used only if it loses at most 3% PlantDoc-val accuracy; it lost 7.7%).

| Model | PlantDoc-val acc | PlantDoc-test acc | Size (MB) | CPU p50 (ms) | CPU p95 (ms) |
|---|---|---|---|---|---|
| EfficientNet-B0 FP32 | 0.611 | 0.542 | 15.4 | 5.0 | 5.4 |
| EfficientNet-B0 INT8 (percentile_conv_nodw) | 0.533 | 0.517 | 4.9 | 7.2 | 8.0 |

CPU: x86_64, 4 threads, batch 1, onnxruntime 1.31.0.

INT8 variants tried (chosen on PlantDoc-val; test not used for the choice):

| Variant | Calibration | Calib. images | Quantized ops | PlantDoc-val acc | Size (MB) | p50 (ms) |
|---|---|---|---|---|---|---|
| minmax_all | MinMax | 500 | all | 0.159 | 4.6 | 5.9 |
| percentile_all | Percentile | 128 | all | 0.424 | 4.6 | 6.2 |
| entropy_all | Entropy | 128 | all | 0.234 | 4.6 | 6.2 |
| minmax_conv | MinMax | 500 | Conv/Gemm/MatMul | 0.185 | 4.6 | 11.4 |
| percentile_conv | Percentile | 128 | Conv/Gemm/MatMul | 0.447 | 4.6 | 7.7 |
| entropy_conv | Entropy | 128 | Conv/Gemm/MatMul | 0.213 | 4.6 | 8.4 |
| percentile_conv_nodw | Percentile | 128 | Conv/Gemm/MatMul, depthwise kept FP32 | 0.533 | 4.9 | 7.4 |
