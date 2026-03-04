# Architecture component-to-file map (Figure 2)

This file maps each block in the paper figure to the exact implementation location in this repo.

## (a) Image Inpainting Pipeline

| Figure component | Where it is implemented | Notes |
|---|---|---|
| Input image `x` and mask `m` | `src/datasets/dataset.py` (`inpaint_dataset.__getitem__`) | Dataset returns `img` and `mask` tensors. |
| Multiply (`x_m = x * m`) | `scripts/train.py` and `scripts/inference.py` | Training/inference both construct masked input this way when mask mode is enabled. |
| DINOv2 (frozen) | `src/dino_cls_token.py`, `src/dinov2/models/vision_transformer.py`, then frozen in `scripts/train.py` / `scripts/inference.py` | `create_model()` builds DINOv2 ViT-B/14 and loads pretrained weights; feature extraction uses `get_feature`. |
| U-Net-like backbone (downsample → residual blocks → upsample) | `src/encoder/resnet_lama.py` (`GlobalGenerator`) | This is the main image encoder/decoder path used by default configs (`encoder_type='resnet'`). |
| Semantic conditioning into backbone | `src/encoder/resnet_lama.py` (`ResnetBlock`) | DINO cls token is mapped to `(alpha, beta, gamma)` and modulates normalized features. |
| Patch-level Gaussian prediction head | `src/feature_map/direct_map.py` (`DirectMap.map`) | Produces Gaussian parameters per patch: color, covariance (cholesky), and offsets/means. |
| Differentiable rasterization | `src/gaussian_kernel_bigger_patch.py` (+ CUDA extension under `src/gaussian_cuda/`) | Projects and rasterizes per-patch Gaussians into output image. |
| Optional overlap composition | `src/utils/overlap.py` and usage in `src/gaussian.py` | Used when `overlap=True` to blend enlarged patch outputs. |
| Top-level model assembly | `src/gaussian.py` (`patch_gaussian`) | Connects encoder → parameter map → rasterization into final prediction. |
| Output image `x_hat` | `src/gaussian.py` (`patch_gaussian.forward_base`) | Returns rasterized image tensor (`pred`). |
| Reconstruction loss `L_recons` | `scripts/train.py` (`Trainer.update`) | Full or masked reconstruction controlled by config. |
| Alignment loss `L_align` | `scripts/train.py` (`cosine_similarity`, `Trainer.update`) | Implemented as `dino_pred_loss` between mapped predicted DINO feature and clean-image DINO feature. |

## (b) SA ResNet block details

| Figure sub-block | Where it is implemented | Notes |
|---|---|---|
| DINO feature input | `scripts/train.py` / `scripts/inference.py` calling `get_dino_feature(...)` | Input to DINO is `img*mask` (or `img` in unmasked inference branch). |
| "semantic map" + MLP path | `src/encoder/resnet_lama.py` (`ResnetBlock.adapter_mlp`) | No separately named `semantic map` module exists; effect is represented by linear projection from cls token to modulation params. |
| `alpha`, `beta` for scale/shift | `src/encoder/resnet_lama.py` (`ResnetBlock.forward`) | `x = x*(1+alpha) + beta` after feature normalization. |
| `gamma` scale gate | `src/encoder/resnet_lama.py` (`ResnetBlock.forward`) | Applied after conv block as `x = x * gamma`. |
| norm | `src/encoder/resnet_lama.py` (`calc_mean_std` + normalization in `ResnetBlock.forward`) | Uses AdaIN-style mean/std normalization. |
| conv-norm-act ×2 | `src/encoder/resnet_lama.py` (`ResnetBlock.build_conv_block`) | Two conv stages with norm/activation pattern. |
| residual add | `src/encoder/resnet_lama.py` (`out = x + before_x`) | Standard residual connection. |

## (c) Patch Level Rasterization

| Figure component | Where it is implemented | Notes |
|---|---|---|
| Patch-level feature tensor | `src/gaussian.py` (`feat = self.encoder(...)`) | Feature tokens per patch from encoder. |
| Mean map | `src/feature_map/direct_map.py` (`mlp_xy` and `offset`) | Mean offsets are predicted and converted to Gaussian centers in `get_iter`. |
| Covariance map | `src/feature_map/direct_map.py` (`cholesky_linear`, `mlp_gaudict`, `self.cholesky`) | Covariance/cholesky parameters are generated from feature vectors. |
| Color map | `src/feature_map/direct_map.py` (`color_mlp`, `self.color`) | RGB coefficients per Gaussian. |
| Gaussian coefficient + composition | `src/gaussian_kernel_bigger_patch.py` | `project_bigger_patch_gs` + `rasterize_bigger_patch_gs` compose final output image. |

## Entry files (run flow)

- Training loop and losses: `scripts/train.py`
- Inference loop: `scripts/inference.py`
- Model config defaults: `configs/train_cfg.py`, `configs/inference_cfg.py`
