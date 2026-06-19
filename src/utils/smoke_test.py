"""Quick environment smoke test: verifies torch+CUDA, our model factory, a train/eval
forward pass for each detector, torchmetrics mAP, and the augmentation/ensemble libs.

    python src/utils/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.build_model import ARCHS, build_model           # noqa: E402
from src.utils.coco_dataset import NUM_CLASSES                  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] torch {torch.__version__} | cuda={torch.cuda.is_available()} "
          f"| device={torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}")

    # dummy image + target (2 boxes)
    img = torch.rand(3, 320, 320)
    target = {"boxes": torch.tensor([[10., 10., 100., 100.], [50., 60., 200., 220.]]),
              "labels": torch.tensor([1, 2])}

    for arch in ARCHS:
        model = build_model(arch, NUM_CLASSES, pretrained=True).to(dev)
        model.train()
        loss_dict = model([img.to(dev)], [{k: v.to(dev) for k, v in target.items()}])
        loss = sum(loss_dict.values())
        model.eval()
        with torch.inference_mode():
            out = model([img.to(dev)])[0]
        print(f"[2] {arch:<22} train_loss={loss.item():.3f} "
              f"eval_dets={len(out['boxes'])}  OK")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # torchmetrics mAP
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    m = MeanAveragePrecision(box_format="xyxy")
    m.update([{"boxes": target["boxes"], "scores": torch.tensor([0.9, 0.8]),
               "labels": target["labels"]}],
             [{"boxes": target["boxes"], "labels": target["labels"]}])
    print(f"[3] torchmetrics mAP OK (map={float(m.compute()['map']):.3f})")

    # augmentation + ensemble libs
    import albumentations  # noqa: F401
    from ensemble_boxes import weighted_boxes_fusion  # noqa: F401
    print("[4] albumentations + ensemble_boxes import OK")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
