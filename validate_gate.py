"""Validate the spike-and-slab inference gate fix: after 'pruning' (setting a
fraction of pruning_parameter < 0), the eval-time quantized weight must contain
that fraction of EXACT zeros. Before the fix it was ~0% (NZ ratio 1.0)."""
import torch
import SQS.config as cfg
import SQS.modeling  # noqa: resolves circular import
from SQS.modeling.DGMS.GMM import gmm_approximation

cfg.METHOD = "SQS"; cfg.PRUNE = True; cfg.PRIOR = "spike_slab"
cfg.IS_NORMAL = False; cfg.SAMPLE = False; cfg.K_LEVEL = 16
cfg.TAU = 0.001; cfg.PRUNE_SCALE = 0.01

w = (torch.randn(8192, device="cuda") * 0.02).contiguous()
gmm = gmm_approximation(16, w, 0.001, init_method="k-means", sigma=3)

for target in (0.0, 0.3, 0.5, 0.75):
    n = w.numel()
    k = int(target * n)
    gmm.pruning_parameter.data.fill_(0.05)      # kept -> soft gate ~1
    gmm.pruning_parameter.data[:k] = -0.1        # pruned -> soft gate ~0
    # The eval gate keys off the pruner's explicit mask (set by generate_mask in the
    # real pipeline), so the test must set it too: mask=True means pruned.
    gmm.mask = torch.zeros_like(gmm.pruning_parameter, dtype=torch.bool)
    gmm.mask[:k] = True
    out = gmm(weights=w, train=False)
    realized = (out == 0).float().mean().item()
    print(f"[GATE] target_pruned={target:.2f}  realized_exact_zero={realized:.4f}")

assert (gmm(weights=w, train=False) == 0).float().mean().item() > 0.7, "gate not realizing sparsity"
print("GATE_OK")
