import os, torch
from torchvision import datasets
root = os.environ["SQS_DATA_DIR"]
datasets.CIFAR10(root=root, train=True, download=True)
datasets.CIFAR10(root=root, train=False, download=True)
print("[OK] CIFAR-10 dataset", flush=True)
for mid in ["cifar10_resnet56", "cifar10_resnet20", "cifar10_resnet32"]:
    m = torch.hub.load("chenyaofo/pytorch-cifar-models", mid, pretrained=True)
    print(f"[OK] {mid}: {sum(p.numel() for p in m.parameters())/1e6:.3f}M params", flush=True)
print("PREFETCH_CIFAR10 DONE", flush=True)
