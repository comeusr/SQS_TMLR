"""Fair effective-bits accounting for SQS vs AWQ (rebuttal 2).

The paper reports SQS compression as  32 / log2(K) * 1/NonZero  — which counts ONLY
the cluster-index bits and IGNORES the sparsity mask, the codebook, and the fp16
outlier window. AWQ's 4-bit number, if you count its per-group scale+zero-point, is
~4.16 bits/weight. To compare fairly we must count the SAME categories of metadata
for BOTH methods. This module does that and prints effective bits/weight + true
compression for each, so the rebuttal can state accuracy-at-equal-effective-bits.

Run: python effective_bits.py            # prints the SST-2 Qwen table
"""
import math


def sqs_effective_bits(K, nonzero_frac, layer_numel, out_features,
                       outlier_first_n=64, outlier_last_n=64,
                       mask_bits_per_weight=1.0, codebook_fp=16):
    """Effective bits/weight for SQS = spike-and-slab pruning + K-cluster quantization.

    Components (all amortized to per-weight of the FULL tensor):
      - cluster index:   nonzero_frac * log2(K)     (only kept weights carry an index)
      - sparsity mask:   mask_bits_per_weight        (1.0 = dense bitmap; the honest floor)
      - codebook:        K * codebook_fp / layer_numel   (K fp16 centroids per layer)
      - outlier window:  (first_n+last_n) fp16 stored verbatim, per attention proj row-set
    """
    index = nonzero_frac * math.log2(K)
    mask = mask_bits_per_weight
    codebook = K * codebook_fp / layer_numel
    n_outlier = (outlier_first_n + outlier_last_n)
    outlier = n_outlier * codebook_fp / layer_numel
    total = index + mask + codebook + outlier
    return {"index": index, "mask": mask, "codebook": codebook,
            "outlier": outlier, "bits_per_weight": total,
            "compression": 32.0 / total}


def sqs_paper_bits(K, nonzero_frac):
    """The paper's OPTIMISTIC number: ignores mask/codebook/outlier entirely."""
    b = nonzero_frac * math.log2(K)
    return {"bits_per_weight": b, "compression": 32.0 / b,
            "paper_formula": f"32/log2({K}) * 1/{nonzero_frac} = {32/math.log2(K)/nonzero_frac:.2f}x"}


def awq_effective_bits(w_bit, group_size=128, scale_fp=16):
    """Effective bits/weight for AWQ = group-wise INT-w_bit + per-group scale & zero-point.
    Overhead per group of `group_size`: one fp16 scale + one w_bit zero-point."""
    overhead = (scale_fp + w_bit) / group_size
    total = w_bit + overhead
    return {"weight": w_bit, "group_overhead": overhead,
            "bits_per_weight": total, "compression": 32.0 / total}


def _row(name, d):
    comp = d.get("compression", 0)
    print(f"  {name:34s} {d['bits_per_weight']:6.3f} b/w   {comp:6.2f}x")


if __name__ == "__main__":
    # Qwen2.5-0.5B, a representative mid-network projection (used for codebook amortization).
    # up_proj in Qwen2.5-0.5B: hidden 896 -> intermediate 4864, numel ~ 4.36M, out=4864.
    LAYER_NUMEL, OUT = 896 * 4864, 4864

    print("=" * 64)
    print("FAIR EFFECTIVE-BITS ACCOUNTING  (Qwen2.5-0.5B / SST-2)")
    print("=" * 64)
    print("\nSQS  K=64 (6-bit codebook), 50% pruned:")
    _row("paper formula (index only)", sqs_paper_bits(64, 0.5))
    _row("honest (+mask+codebook+outlier)", sqs_effective_bits(64, 0.5, LAYER_NUMEL, OUT))

    print("\nSQS  K=16 (4-bit codebook), 50% pruned:")
    _row("paper formula (index only)", sqs_paper_bits(16, 0.5))
    _row("honest", sqs_effective_bits(16, 0.5, LAYER_NUMEL, OUT))

    print("\nAWQ (group-wise INT4, g=128, fp16 scale + int zp):")
    _row("4-bit", awq_effective_bits(4, 128))
    _row("3-bit", awq_effective_bits(3, 128))

    print("\n--> Fair comparison point: SQS honest bits/weight vs AWQ bits/weight.")
    s = sqs_effective_bits(64, 0.5, LAYER_NUMEL, OUT)["bits_per_weight"]
    a = awq_effective_bits(4, 128)["bits_per_weight"]
    print(f"    SQS(K=64,50%) = {s:.3f} b/w   AWQ(4-bit) = {a:.3f} b/w")
    print(f"    => compare ACCURACY at ~{(s+a)/2:.1f} effective bits/weight.")
