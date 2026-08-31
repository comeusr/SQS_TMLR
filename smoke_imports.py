import importlib, sys, traceback

def check(label, fn):
    try:
        fn()
        print(f"[OK]   {label}")
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")
        traceback.print_exc()

# Core stack
def _torch():
    import torch, numpy
    print("      torch", torch.__version__, "numpy", numpy.__version__)
check("torch+numpy", _torch)

check("composer (mosaicml)", lambda: importlib.import_module("composer"))
check("transformers", lambda: importlib.import_module("transformers"))

# transformers symbols the repo imports (version-sensitive)
def _qwen_syms():
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention, Qwen2MLP, apply_rotary_pos_emb, repeat_kv,
        Qwen2RotaryEmbedding, eager_attention_forward)
check("qwen2 symbols (apply_rotary/eager_attention_forward/...)", _qwen_syms)

def _misc_tf():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward, FlashAttentionKwargs
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.processing_utils import Unpack
check("transformers flash/attention utils", _misc_tf)

# SQS package (installed editable)
check("SQS.config", lambda: importlib.import_module("SQS.config"))
check("SQS.modeling (DGMSNet)", lambda: __import__("SQS.modeling", fromlist=["DGMSNet"]).DGMSNet)
check("SQS.modeling.DGMS (DGMSConv)", lambda: __import__("SQS.modeling.DGMS", fromlist=["DGMSConv"]).DGMSConv)
check("SQS.QuantAttention (Qwen custom)", lambda: __import__("SQS.QuantAttention", fromlist=["CustomizedQwen2Attention"]).CustomizedQwen2Attention)
check("SQS.utils.GPT2_pruner_quantizer", lambda: importlib.import_module("SQS.utils.GPT2_pruner_quantizer"))
check("SQS.utils.algorithm (GMM_Pruning)", lambda: __import__("SQS.utils.algorithm", fromlist=["GMM_Pruning"]).GMM_Pruning)
check("SQS.utils.misc (get_device)", lambda: __import__("SQS.utils.misc", fromlist=["get_device"]).get_device)
check("SQS.utils.watch (EpochMonitor)", lambda: __import__("SQS.utils.watch", fromlist=["EpochMonitor"]).EpochMonitor)

# ResNet-side third-party
check("timm", lambda: importlib.import_module("timm"))
check("detectors (registers cifar models)", lambda: importlib.import_module("detectors"))

def _cifar_model_registered():
    import timm
    # detectors registers these names; just check the entry exists (no download)
    names = timm.list_models("*cifar100*")
    print("      cifar100 models available:", len(names), names[:6])
    assert any("resnet" in n for n in names), "no resnet cifar100 model registered"
check("timm cifar100 model registry", _cifar_model_registered)

# bitsandbytes (may need GPU; import only)
def _bnb():
    import bitsandbytes as bnb
check("bitsandbytes import", _bnb)

print("\nDONE")
