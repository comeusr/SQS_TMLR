"""HF-hub assets only (fast via hf_transfer) — runs in parallel with the slow
torchvision CIFAR download."""
import os, traceback

def step(label, fn):
    try:
        fn(); print(f"[OK]   {label}", flush=True)
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

def _timm():
    import detectors, timm
    m = timm.create_model("resnet18_cifar100", pretrained=True)
    print("      resnet18_cifar100 params:",
          sum(p.numel() for p in m.parameters())/1e6, "M")
step("timm resnet18_cifar100 pretrained", _timm)

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

def _glue():
    from datasets import load_dataset
    d = load_dataset("glue", "sst2", trust_remote_code=True)
    print("      sst2 splits:", {k: len(v) for k, v in d.items()})
step("GLUE sst2 dataset", _glue)

print("PREFETCH_HF DONE", flush=True)
