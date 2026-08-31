"""Activation-aware pruning saliency (Wanda-style) for SQS.

SQS ranks weights for pruning by pure magnitude (|quantized_w| * sigmoid(gate)).
Wanda (Sun et al., ICLR 2024) shows |W_ij| * ||X_j||_2 — weight magnitude scaled by
the L2 norm of the corresponding input activation channel — is far stronger for LLMs,
matching Hessian-based SparseGPT at 50% sparsity with no weight updates.

This module:
  1) collect_activation_norms: one calibration pass over the *base* model (before the
     SQS layer-replacement/sort) to get per-input-channel ||X_j||_2 for every projection.
  2) assign_activation_norms: after SQS_INIT (which flattens+sorts each projection's
     weight and stores argsorted_*_indices), broadcasts ||X_j|| to every weight element,
     re-orders it through the SAME sort permutation, slices it per GMM block, and stores
     it on each block as `.act_norm`.

GMM.forward then multiplies sweight_cache by `self.act_norm` so the pruner's saliency
becomes |quantized_w| * sigmoid(gate) * ||X_j||  — activation-aware.
"""
import torch
import torch.nn as nn

_PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@torch.no_grad()
def collect_activation_norms(model, calib_loader, n_batches, device):
    """Return {module_name: ||X_j||_2 tensor [in_features]} for each projection Linear,
    measured on the base model's standard forward (self.q_proj(x) etc. fire hooks)."""
    sums, counts, handles = {}, {}, []

    def mk_hook(name):
        def pre_hook(module, args):
            x = args[0]
            x = x.reshape(-1, x.shape[-1]).float()          # [tokens, in]
            s = sums.get(name)
            sums[name] = x.pow(2).sum(dim=0) if s is None else s + x.pow(2).sum(dim=0)
            counts[name] = counts.get(name, 0) + x.shape[0]
        return pre_hook

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.endswith(_PROJ_SUFFIXES) and "score" not in name:
            handles.append(m.register_forward_pre_hook(mk_hook(name)))

    model.eval()
    seen = 0
    for batch in calib_loader:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        model(**{k: v for k, v in batch.items() if k != "labels"})
        seen += 1
        if seen >= n_batches:
            break
    for h in handles:
        h.remove()

    # ||X_j||_2 over the calibration tokens (same token count per layer -> comparable).
    return {name: torch.sqrt(sums[name]) for name in sums}


@torch.no_grad()
def _sorted_elementwise_actnorm(act_norm_channel, out_features, argsorted_indices, device):
    """Broadcast per-channel ||X|| to [out*in] (row-major), then permute into the
    projection's sorted-weight order. sort_perm = argsort(argsorted_indices) because for a
    permutation P, argsort(argsort(P)) = P."""
    in_features = act_norm_channel.numel()
    act_flat = act_norm_channel.to(device).repeat(out_features)          # [out*in], [r*in+c]=norm[c]
    sort_perm = torch.argsort(argsorted_indices.to(device))              # recovers the sort permutation
    return act_flat[sort_perm]                                           # aligned to sorted weight


@torch.no_grad()
def assign_activation_norms(model, act_norms, device):
    """Attach per-block `.act_norm` (in sorted order) to every non-Identity GMM sub-block."""
    # Import here to avoid a circular import at module load.
    from SQS.QuantAttention import CustomizedQwen2Attention, CustomizedQwen2MLP, \
        CustomizedLlamaAttention, CustomizedLLamaMLP

    n_assigned = 0
    for name, m in model.named_modules():
        if isinstance(m, (CustomizedQwen2Attention, CustomizedLlamaAttention)):
            fn, ln = m.first_n, m.last_n
            projs = [("q_proj", m.q_proj, getattr(m, "argsorted_q_proj_indices", None), fn, ln),
                     ("k_proj", m.k_proj, getattr(m, "argsorted_k_proj_indices", None), fn, ln),
                     ("v_proj", m.v_proj, getattr(m, "argsorted_v_proj_indices", None), fn, ln),
                     ("o_proj", m.o_proj, getattr(m, "argsorted_o_proj_indices", None), fn, ln)]
        elif isinstance(m, (CustomizedQwen2MLP, CustomizedLLamaMLP)):
            projs = [("up_proj", m.up_proj, getattr(m, "up_argsort_indices", None), m.up_first_n, m.up_last_n),
                     ("down_proj", m.down_proj, getattr(m, "down_argsort_indices", None), m.down_first_n, m.down_last_n)]
        else:
            continue

        for pname, proj, argsorted, first_n, last_n in projs:
            if argsorted is None:
                continue
            full = name + "." + pname
            if full not in act_norms:
                continue
            numel = act_norms[full].numel() * (proj.weight.numel() // act_norms[full].numel())
            out_features = proj.weight.numel() // act_norms[full].numel()
            act_sorted = _sorted_elementwise_actnorm(
                act_norms[full], out_features, argsorted, device)
            # Per-layer mean-normalization is ESSENTIAL: SQS ranks pruning saliency
            # with a single GLOBAL threshold (torch.kthvalue over all layers' blocks
            # concatenated). Raw ||X_j|| differs by orders of magnitude across layers,
            # so injecting it directly makes large-norm layers keep ~everything and
            # small-norm layers get gutted -> sparsity mis-allocated -> accuracy drops.
            # Dividing by the layer mean keeps each layer's total saliency mass the same
            # as the magnitude-only baseline, so Wanda only RE-RANKS *within* a layer
            # (boost high-activation channels, demote low) -- the intended improvement.
            act_sorted = act_sorted / (act_sorted.mean() + 1e-8)
            # Walk the block list in sorted order, skipping the Identity outlier blocks
            # (block 0 has length first_n, the last block has length last_n).
            offset = 0
            n_blocks = len(proj.sub_distribution_list)
            for idx, sub in enumerate(proj.sub_distribution_list):
                if isinstance(sub, nn.Identity) or not hasattr(sub, "pruning_parameter"):
                    offset += first_n if idx == 0 else last_n
                    continue
                n = sub.pruning_parameter.numel()
                sub.act_norm = act_sorted[offset:offset + n].reshape_as(sub.pruning_parameter)
                offset += n
                n_assigned += 1
    return n_assigned
