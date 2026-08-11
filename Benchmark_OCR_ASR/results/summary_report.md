# OCR & ASR Benchmark Summary — L29

Generated with Python 3.12.0 on Windows-11-10.0.26200-SP0.

Ground truth approved: OCR=True, ASR=True.

## OCR

| model_id            | text_type       |   samples |   cer_norm |   cer_ci_low |   cer_ci_high |   character_accuracy |   wer_norm |   exact_norm |   detection_recall_iou50 |   latency_median_sec |   latency_p95_sec |   vram_peak_gb |   false_positive_rate |
|:--------------------|:----------------|----------:|-----------:|-------------:|--------------:|---------------------:|-----------:|-------------:|-------------------------:|---------------------:|------------------:|---------------:|----------------------:|
| paddleocr_v5_mobile | dynamic_overlay |        37 |     0.7003 |       0.5811 |        0.8246 |               0.2997 |     1.0261 |       0      |                   0.7568 |               0.0494 |            0.06   |         1.2603 |                 nan   |
| paddleocr_v5_mobile | no_text         |        10 |     0.9    |       0.7    |        1      |               0.1    |     0.9    |       0.1    |                   0      |               0.0457 |            0.0498 |         1.2603 |                   0.9 |
| paddleocr_v5_mobile | static_text     |        13 |     0.4441 |       0.1679 |        0.7897 |               0.5559 |     0.7681 |       0.1538 |                   0.9231 |               0.0527 |            0.0722 |         1.2603 |                 nan   |
| easyocr_det_vietocr | dynamic_overlay |        37 |     0.5802 |       0.4651 |        0.7122 |               0.4198 |     0.877  |       0.0811 |                   0.973  |               0.3431 |            0.5221 |         0.7537 |                 nan   |
| easyocr_det_vietocr | no_text         |        10 |     0.8    |       0.5    |        1      |               0.2    |     0.8    |       0.2    |                   0      |               0.2378 |            0.3196 |         0.7537 |                   0.8 |
| easyocr_det_vietocr | static_text     |        13 |     0.4877 |       0.069  |        1.087  |               0.5123 |     0.4    |       0.3846 |                   1      |               0.4593 |            1.0615 |         0.7537 |                 nan   |
| easyocr_vi_en       | dynamic_overlay |        37 |     0.5947 |       0.4852 |        0.7084 |               0.4053 |     0.8432 |       0.0811 |                   0.973  |               0.1889 |            0.2394 |         0.6111 |                 nan   |
| easyocr_vi_en       | no_text         |        10 |     0.8    |       0.5    |        1      |               0.2    |     0.8    |       0.2    |                   0      |               0.168  |            0.2128 |         0.6111 |                   0.8 |
| easyocr_vi_en       | static_text     |        13 |     0.24   |       0.1397 |        0.347  |               0.76   |     0.6637 |       0.1538 |                   1      |               0.2014 |            0.3147 |         0.6111 |                 nan   |

## ASR

| model_id          |   clips |   wer_norm |   wer_ci_low |   wer_ci_high |   cer_norm |   timestamp_valid_rate |   timestamp_coverage |   rtf_median |   rtf_p95 |   vram_peak_gb |
|:------------------|--------:|-----------:|-------------:|--------------:|-----------:|-----------------------:|---------------------:|-------------:|----------:|---------------:|
| phowhisper_small  |      20 |     0.4332 |       0.2577 |        0.6656 |     0.376  |                    1   |               0.9998 |       0.1281 |    0.2892 |         0.8351 |
| phowhisper_medium |      20 |     0.5528 |       0.3044 |        0.9189 |     0.4695 |                    1   |               0.9254 |       0.2234 |    0.5267 |         2.3568 |
| whisper_small     |      20 |     0.6656 |       0.425  |        0.954  |     0.4902 |                    0.8 |               0.8773 |       0.1221 |    0.3942 |         0.8352 |

## Machine-readable conclusion

```yaml
conclusion:
  ocr:
    selected_model: null
    reason: no OCR model meets static character accuracy threshold
  asr:
    selected_model: phowhisper_small
  selection_policy: accuracy first; latency then VRAM for ties
  ground_truth_approved:
    ocr: true
    asr: true
```
