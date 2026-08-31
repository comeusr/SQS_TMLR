"""Isolate the ResNet prune-onset collapse. Runs on CPU with a small test subset.

Questions:
  (1) Does the pretrained resnet18_cifar100 actually load (base ~79%)?
  (2) Does GLOBAL 50% magnitude pruning collapse it (=> SQS needs per-layer thresh)?
  (3) Does PER-LAYER 50% magnitude pruning preserve it (=> the fix)?
If (2) collapses and (3) is fine, the ResNet bug is the SAME global-threshold issue
we just fixed for Qwen Wanda -- SQS's single global kthvalue mis-allocates sparsity
across conv layers of wildly different scale.
"""
import os, torch, torch.nn as nn
import detectors  # noqa: registers resnet18_cifar100
import timm
import torchvision, torchvision.transforms as T

os.environ.setdefault("HF_HUB_OFFLINE", "1")
dev = "cpu"

def get_loader(n=1000):
    # CIFAR-100 test, timm resnet18_cifar100 normalization (standard CIFAR stats).
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    root = os.environ.get("CIFAR_ROOT", "/anvil/scratch/x-ashyam2/.cache/cifar")
    ds = torchvision.datasets.CIFAR100(root, train=False, download=False, transform=tf)
    ds = torch.utils.data.Subset(ds, list(range(n)))
    return torch.utils.data.DataLoader(ds, batch_size=200, num_workers=2)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = tot = 0
    for x, y in loader:
        p = model(x).argmax(1)
        correct += (p == y).sum().item(); tot += y.numel()
    return correct / tot

def conv_weights(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and m.weight.numel() > 1000]

@torch.no_grad()
def prune_global(model, sparsity):
    allw = torch.cat([m.weight.abs().flatten() for _, m in conv_weights(model)])
    thr = torch.kthvalue(allw, int(sparsity * allw.numel()))[0]
    for _, m in conv_weights(model):
        m.weight.mul_((m.weight.abs() >= thr).float())

@torch.no_grad()
def prune_per_layer(model, sparsity):
    for _, m in conv_weights(model):
        w = m.weight.abs().flatten()
        thr = torch.kthvalue(w, int(sparsity * w.numel()))[0]
        m.weight.mul_((m.weight.abs() >= thr).float())

if __name__ == "__main__":
    loader = get_loader()
    base = timm.create_model("resnet18_cifar100", pretrained=True).to(dev)
    print(f"(1) base pretrained acc          = {evaluate(base, loader):.4f}")

    import copy
    mg = copy.deepcopy(base); prune_global(mg, 0.50)
    print(f"(2) GLOBAL 50% magnitude prune   = {evaluate(mg, loader):.4f}")

    mp = copy.deepcopy(base); prune_per_layer(mp, 0.50)
    print(f"(3) PER-LAYER 50% magnitude prune= {evaluate(mp, loader):.4f}")
