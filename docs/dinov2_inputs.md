# DINOv2 inputs in this repository

This repo feeds DINOv2 from `get_dino_feature(...)` in `src/dino_cls_token.py`. The actual tensors passed into that call are created in two places:

1. `scripts/train.py`
   - masked input path: `input = img * mask`, then `dino_feature = get_dino_feature(input, self.dino_model)`
   - optional clean-image branch for prediction loss: `dino_clean_feature = get_dino_feature(img, self.dino_model)`

2. `scripts/inference.py`
   - masked input path: `input = img * mask`, then `dino_feature = get_dino_feature(input, self.dino_model)`
   - unmasked path: `input = img`, then `dino_feature = get_dino_feature(input, self.dino_model)`

## Extraction utility

Use `scripts/extract_dinov2_inputs.py` to export exactly the tensors that are sent to DINOv2:

```bash
python scripts/extract_dinov2_inputs.py \
  --cfg configs/inference_cfg.py \
  --mode inference \
  --output-dir outputs/dinov2_inputs \
  --num-batches 1
```

For training configs, if `use_dino_pred_loss=True` and masks are enabled, the script also exports the clean-image DINO input (`*_dinov2_clean_input.pt`).
