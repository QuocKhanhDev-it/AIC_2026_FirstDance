# OCR & ASR Benchmark Summary — L29

Generated with Python 3.12.0 on Windows-11-10.0.26200-SP0.

Ground truth approved: OCR=False, ASR=False.

## OCR

| model_id            | text_type   |   samples |   cer_norm |   cer_ci_low |   cer_ci_high |   character_accuracy |   wer_norm |   exact_norm |   detection_recall_iou50 |   latency_median_sec |   latency_p95_sec |   vram_peak_gb |
|:--------------------|:------------|----------:|-----------:|-------------:|--------------:|---------------------:|-----------:|-------------:|-------------------------:|---------------------:|------------------:|---------------:|
| paddleocr_v5_mobile | static_text |         1 |     0.1667 |       0.1667 |        0.1667 |               0.8333 |          1 |            0 |                        1 |               0.1983 |            0.1983 |         2.0443 |
| easyocr_det_vietocr | static_text |         1 |     0      |       0      |        0      |               1      |          0 |            1 |                        1 |               0.3281 |            0.3281 |         0.7537 |
| easyocr_vi_en       | static_text |         1 |     0.1667 |       0.1667 |        0.1667 |               0.8333 |          1 |            0 |                        1 |               0.1806 |            0.1806 |         0.6111 |

## ASR

| model_id          |   clips |   wer_norm |   wer_ci_low |   wer_ci_high |   cer_norm |   timestamp_valid_rate |   timestamp_coverage |   rtf_median |   rtf_p95 |   vram_peak_gb |
|:------------------|--------:|-----------:|-------------:|--------------:|-----------:|-----------------------:|---------------------:|-------------:|----------:|---------------:|
| phowhisper_small  |       1 |     0      |       0      |        0      |     0      |                      1 |               1      |       0.0703 |    0.0703 |         0.7765 |
| phowhisper_medium |       1 |     1.4211 |       1.4211 |        1.4211 |     1.4789 |                      1 |               1      |       0.5154 |    0.5154 |         2.3568 |
| whisper_small     |       1 |     0.193  |       0.193  |        0.193  |     0.0651 |                      0 |               0.6333 |       0.0769 |    0.0769 |         0.7744 |

## Machine-readable conclusion

```yaml
conclusion:
  ocr:
    selected_model: null
    reason: OCR ground truth is not fully approved
  asr:
    selected_model: null
    reason: ASR ground truth is not fully approved
  selection_policy: accuracy first; latency then VRAM for ties
  ground_truth_approved:
    ocr: false
    asr: false
```
