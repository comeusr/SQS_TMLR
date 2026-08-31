"""Pre-download all datasets/weights on the login node (has internet) so the
GPU job can run fully offline."""
import os, sys, traceback

def step(label, fn):
    try:
        fn(); print(f"[OK]   {label}", flush=True)
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

# CIFAR-100 (torchvision)
def _cifar():
    from torchvision import datasets
    root = os.environ["SQS_DATA_DIR"]
    datasets.CIFAR100(root=root, train=True, download=True)
    datasets.CIFAR100(root=root, train=False, download=True)
step("CIFAR-100 dataset", _cifar)

# timm resnet18_cifar100 pretrained weights (registered by `detectors`)
def _timm():
    import detectors, timm
    m = timm.create_model("resnet18_cifar100", pretrained=True)
    print("      resnet18_cifar100 params:",
          sum(p.numel() for p in m.parameters())/1e6, "M")
step("timm resnet18_cifar100 pretrained", _timm)

# Qwen2.5-0.5B tokenizer + model
def _qwen():
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    import torch
    name = "Qwen/Qwen2.5-0.5B"
    AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)
    cfg = AutoConfig.from_pretrained(name, num_labels=2, trust_remote_code=True)
    AutoModelForSequenceClassification.from_pretrained(
        name, config=cfg, attn_implementation="eager",
        torch_dtype=torch.float32, trust_remote_code=True)
step("Qwen2.5-0.5B model+tokenizer", _qwen)

# GLUE sst2
def _glue():
    from datasets import load_dataset
    d = load_dataset("glue", "sst2", trust_remote_code=True)
    print("      sst2 splits:", {k: len(v) for k, v in d.items()})
step("GLUE sst2 dataset", _glue)

print("PREFETCH DONE", flush=True)
