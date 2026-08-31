"""CIFAR-10/100 data loaders for the ResNet quantization experiment.

Reconstructed to satisfy `from dataloader import make_data_loader` in
`src/resnet/resnet_main.py`.  The original DGMS repo shipped a `dataloader`
package (ImageFolder based); it is missing from SQS_private.  This version uses
`torchvision.datasets.CIFAR{10,100}` with `download=True`, so it is fully
self-contained.

Normalization matches the pretrained checkpoints the entry point loads:
  * cifar10  -> chenyaofo/pytorch-cifar-models stats
  * cifar100 -> detectors/timm resnet*_cifar100 stats

Optional smoke-test knobs (no effect unless set):
  SQS_DATA_DIR   directory for the dataset (default: repo Data/)
  SQS_MAX_TRAIN  cap #train samples (Subset) for quick end-to-end checks
  SQS_MAX_EVAL   cap #eval samples
"""
import os

import torch
from torchvision import datasets, transforms

import SQS.config as cfg

# (mean, std) per dataset — must match the pretrained model's preprocessing.
_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def _resolve_root():
    root = os.environ.get(
        "SQS_DATA_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data"),
    )
    os.makedirs(root, exist_ok=True)
    return root


def make_data_loader(args, **kwargs):
    dataset = args.dataset
    if dataset not in _STATS:
        raise NotImplementedError(
            f"make_data_loader was reconstructed for cifar10/cifar100 only, got '{dataset}'."
        )

    data_root = _resolve_root()
    mean, std = _STATS[dataset]

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    DS = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    train_set = DS(root=data_root, train=True, download=True, transform=train_tf)
    test_set = DS(root=data_root, train=False, download=True, transform=test_tf)

    # Optional subsampling for fast end-to-end smoke tests.
    max_train = int(os.environ.get("SQS_MAX_TRAIN", "0"))
    max_eval = int(os.environ.get("SQS_MAX_EVAL", "0"))
    if max_train > 0:
        train_set = torch.utils.data.Subset(train_set, range(min(max_train, len(train_set))))
    if max_eval > 0:
        test_set = torch.utils.data.Subset(test_set, range(min(max_eval, len(test_set))))

    batch_size = getattr(args, "batch_size", None) or cfg.BATCH_SIZE.get(dataset, 128)
    workers = getattr(args, "workers", 4)
    nclass = cfg.NUM_CLASSES[dataset]

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    # DGMS uses the same held-out set for validation and test.
    val_loader = test_loader
    return train_loader, val_loader, test_loader, nclass
