"""CPU dry-run of the ResNet setup path (mirrors resnet_main.py up to, but not
including, the CUDA-only k-means init). Validates imports, model load, DGMSNet
wrap, and the Conv2d->DGMSConv TorchTransformer replacement."""
import argparse
import torch.nn as nn
import timm, detectors  # detectors registers resnet*_cifar100

import SQS.config as cfg
from SQS.modeling import DGMSNet
from SQS.modeling.DGMS import DGMSConv
from SQS.utils.PyTransformer.transformers.torchTransformer import TorchTransformer

args = argparse.Namespace(
    num_classes=100, freeze_bn=False, empirical=False, normal=False,
    tau=0.001, K=16, init_method='k-means', method='SQS', prune_scale=0.01,
    debug=False, sample=True, average=False, average_num=2,
    prior='spike_slab', freeze_weight=True,
)
cfg.set_config(args)
print("after set_config: PRUNE=%s METHOD=%s IS_NORMAL=%s K=%s" %
      (cfg.PRUNE, cfg.METHOD, cfg.IS_NORMAL, cfg.K_LEVEL))
assert cfg.PRUNE is True and cfg.METHOD == "SQS", "set_config didn't pick up SQS"

model = timm.create_model("resnet18_cifar100", pretrained=True)
n_conv_before = sum(isinstance(m, nn.Conv2d) for m in model.modules())

model = DGMSNet(model, args, False)
cfg.IS_NORMAL = False

t = TorchTransformer()
t.register(nn.Conv2d, DGMSConv)
model = t.trans_layers(model)

n_dgms = sum(isinstance(m, DGMSConv) for m in model.modules())
n_conv_left = sum(1 for m in model.modules()
                  if isinstance(m, nn.Conv2d) and not isinstance(m, DGMSConv))
print(f"conv2d before wrap: {n_conv_before}")
print(f"DGMSConv after transform: {n_dgms}   plain Conv2d left: {n_conv_left}")
assert n_dgms > 0, "no convs were converted to DGMSConv"
print("total params: %.2fM" % (sum(p.numel() for p in model.parameters())/1e6))
print("RESNET_CPU_OK")
