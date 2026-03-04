import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader
from mmengine import Config

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)



def get_dino_inputs(img: torch.Tensor, mask: torch.Tensor, enable_mask: bool):
    """Replicates the DINO input construction used by train/inference scripts."""
    if enable_mask:
        return img * mask
    return img


def save_tensor_batch(tensor: torch.Tensor, paths, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(tensor.shape[0]):
        name = os.path.splitext(os.path.basename(paths[idx]))[0]
        save_path = os.path.join(output_dir, f"{name}_{prefix}.pt")
        torch.save(tensor[idx].detach().cpu(), save_path)


def run(cfg_path: str, mode: str, output_dir: str, num_batches: int, device: str, dtype: str):
    from src.datasets.dataset import inpaint_dataset
    from src.utils.util import to_torch_dtype

    cfg = Config.fromfile(cfg_path)
    cfg_dict = cfg.to_dict()

    if mode == "train":
        run_param = cfg_dict["train_param"]
        model_param = cfg_dict["model_param"]
        use_dino_pred_loss = model_param.get("use_dino_pred_loss", False)
    else:
        run_param = cfg_dict["inference_param"]
        use_dino_pred_loss = False

    enable_mask = run_param["mask"]

    dataset_param = dict(cfg_dict["data_param"])
    shuffle = dataset_param.pop("shuffle")
    num_workers = dataset_param.pop("num_workers")
    batch_size = dataset_param.pop("bs")

    dataset = inpaint_dataset(**dataset_param)
    dataloader = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    torch_device = torch.device(device)
    torch_dtype = to_torch_dtype(dtype)

    for iter_idx, batch in enumerate(dataloader):
        if num_batches > 0 and iter_idx >= num_batches:
            break

        img = batch["img"].to(torch_device, torch_dtype)
        mask = batch["mask"].to(torch_device, torch_dtype)
        img_path = batch["img_path"]

        dino_input = get_dino_inputs(img, mask, enable_mask)
        save_tensor_batch(dino_input, img_path, output_dir, prefix="dinov2_input")

        if mode == "train" and enable_mask and use_dino_pred_loss:
            save_tensor_batch(img, img_path, output_dir, prefix="dinov2_clean_input")

    print(f"Saved DINOv2 inputs to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract tensors that are fed to DINOv2 in this repo.")
    parser.add_argument("--cfg", required=True, help="Path to config file (training or inference config).")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-batches", type=int, default=1, help="How many batches to export. 0 means all.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        cfg_path=args.cfg,
        mode=args.mode,
        output_dir=args.output_dir,
        num_batches=args.num_batches,
        device=args.device,
        dtype=args.dtype,
    )
