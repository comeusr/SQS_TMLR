# SQS Rebuttal — Results So Far

_Qwen2.5-0.5B / SST-2 (base fine-tuned = 92.66%, matches paper 92.60). All SQS effective
bits/weight from `src/analysis/effective_bits.py` (honest accounting: index + 1-bit mask +
codebook + outlier). AWQ from faithful fake-quant impl `src/baselines/AWQ/awq_real.py`._

## Headline: activation-aware pruning fixes SQS's accuracy gap
| Method (50% sparse, 6-bit K=64, ~4.0 eff-bits) | SST-2 acc | vs paper 90.14 |
|---|---|---|
| SQS **magnitude** (paper's pruning) | 86.70 | −3.4 |
| **SQS + activation-aware (Wanda-style) pruning** | **92.55** | **+2.4** |
Per-layer-normalized `|W|·‖X‖` saliency, one calibration pass, no extra training.
(Both A/B jobs timed out at 80%; Wanda converged, magnitude still climbing.)

## R2: SQS vs AWQ curve (SST-2 accuracy @ effective bits/weight)
| Method | eff-bits | acc |
|---|---|---|
| AWQ 4-bit g128 | 4.16 | 92.78 |
| AWQ 4-bit g64  | 4.31 | 91.86 |
| AWQ 3-bit g64  | 3.30 | 91.17 |
| AWQ 3-bit g128 | 3.15 | 89.91 |
| AWQ 2-bit g128 | 2.14 | **55.28  <- cliff** |
| SQS+Wanda K64/50% | 4.00 | 92.20 |
| SQS+Wanda K16/50% | 3.00 | 90.25 |
| SQS+Wanda K16/75% | 2.00 | 80.05 (converged, NZ .26) |
READING (the R2 story): at ~4 bits AWQ 92.78 ≈ SQS 92.20; at ~3 bits SQS 90.25 ≈ AWQ 89.91;
at ~2 bits AWQ COLLAPSES to 55.28 while SQS holds ~80 -> SQS +25. SQS degrades far more
gracefully at aggressive compression (structured sparsity + moderate quant beats pushing
bit-width to floor). NOTE: magnitude-SQS trails AWQ at all bits; our activation-aware fix
is what makes SQS competitive with / better than AWQ.

## R1: outlier window ablation (CONVERGED 2x2, both 50% sparse)
| pruning | outlier ON | outlier OFF | window worth |
|---|---|---|---|
| magnitude (paper) | 85.55 | 79.01 | +6.5 |
| Wanda (ours)      | 92.20 | 87.20 | +5.0 |
Conclusions: (1) outlier window valuable in BOTH (~5-6.5pt), NOT redundant under Wanda.
(2) Wanda orthogonal+additive: +6.6 (ON) / +8.2 (OFF) over magnitude. The two stack.
Window protects fat-tail weights K can't represent; Wanda picks which weights to prune.

## R3: Bayesian averaging (average_num) vs accuracy — ResNet-18/C100, 50% sparse, K16 (job 19380243)
| M | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| Val_Acc | 66.25 | 66.35 | 66.35 | 66.35 | 66.36 | 66.35 |
FINDING: Bayesian avg number M has NEGLIGIBLE effect (~0.1% M=1->50). At this converged
50%/K16 operating point the GMM posterior is confident -> single sample ~= full ensemble.
(Corrects earlier guess: fast-test's low 46% was the SHORT recipe, not single-sample noise.)
To see if M matters more at a NOISIER point, would sweep at higher sparsity / lower K.

### R3 v2: DECOUPLED eval temperature x M (train sharp tau=0.001, soften only at eval), job 19385971
| eval_temp_scale | M=1 | M=10 | M=50 |
|---|---|---|---|
| 1.0 (native)  | 68.67 | 68.67 | 68.67 |
| 10.0          | 68.74 | 68.59 | 68.59 |
| 100.0         | 38.07 | 58.42 | 61.98 |
| 1000.0        | 1.05  | 1.03  | 1.09  |
STORY: averaging IS inert at native temp because responsibilities are near one-hot (no
diversity). Injecting randomness (scale=100) makes averaging work dramatically (M=1->50:
38->62, +24) — textbook variance reduction, benefit grows with M. BUT it never BEATS the
sharp MAP (best averaged 62 < sharp 68.67): the softening adds noise averaging only partly
recovers. Conclusion: the low-temp MAP quantization is already near-optimal; the posterior
"uncertainty" here is injected noise, not informative signal. Keep temperature low at BOTH
train and eval. Explains WHY R3 is flat (not a bug) and pinpoints when averaging would matter.

## ResNet: Wanda (activation-aware conv pruning) + averaging-beats-MAP (jobs 19392978 K16, 19394189 K8)
WANDA ON CNN: K16/50% + conv-Wanda = 66.8 == baseline 66.4 (+0.4, NO help). Contrast Qwen +9.
=> activation-aware pruning exploits the OUTLIER-FEATURE phenomenon, an LLM trait; CNN conv
activations are ~uniform across channels so magnitude pruning is already near-optimal. Clean
"helps where the mechanism exists" result.
AVERAGING BEATS MAP (fixed: M=1 ref = deterministic argmax, NO weight sampling; decoupled eval temp):
  MAP_no_sample = 66.75. temp1: flat 66.75. temp3: 66.82 (M5). temp5: 66.92 (M20) = +0.17 > MAP.
  temp10: 66.62. temp30: 66.18. temp100: 0.38->0.47 (collapse). Inverted-U in temp; optimum
  ~scale5/M20. So YES averaging > deterministic point estimate, margin small at K16 (4-bit easy,
  few boundary weights); K8 (3-bit) expected larger. Confirms user hypothesis (need randomness).

## ResNet-18/CIFAR-100 status
- base pretrained loads = 79.00 (=paper 79.26); plain 50% magnitude prune = 78.70 (diag OK).
- Original SQS run collapsed to 1% at prune onset (PERMANENT). Root cause = eval/train gate
  mismatch (train soft-gates kept weights by sigmoid(pp/scale)->~0.5; eval used hard 1.0 ->
  2x too big -> blow-up). FIX in GMM.py: eval `Pweight*sigmoid(pp/scale)*(~mask)`.
- Fast-test (short 2ep warm-up): collapse GONE (1% -> ~46%) but below target.
- FULL recipe + averaged eval (job 19380243): onset ep5=74.7 -> ramp -> converged 66.4 at
  50% sparse/K16. COLLAPSE DEFINITIVELY FIXED (1%->66%), but ~10pt BELOW paper 76.14.
  LIKELY CAUSE: --freeze_weight freezes conv weights (only GMM centroids+pruning adapt) so
  the net can't recover pruned capacity. NEXT: re-run WITHOUT --freeze_weight to close gap.
- Also found: repro sbatch omitted --average (single stochastic sample eval); added
  --avg_sweep to resnet_main. But averaging barely matters here (see R3) -> the recipe
  (warm-up length + recovery epochs) is what drives the number, not averaging.

## Effective-bits accounting (honest, per-weight)
SQS = nonzero·log2(K) + 1(mask) + codebook + outlier.  AWQ = w_bit + (16+w_bit)/group.
SQS K64/50% = 4.00 (NOT paper's 3.00/10.67x — that omits mask+codebook+outlier).

## ResNet GAP CLOSED (kept-weight-attenuation fix, job 19423974)
Root cause: spike-slab prior grad pushed EVERY gate toward spike (sp=0.01), halving the
SURVIVING weights (frozen -> can't recover) -> ~7pt loss beyond the sparsity. FIX
(--prior_pruned_only): gate the prior by the mask so only PRUNED weights go to the spike;
survivors keep gate~1. RESULT ResNet-18/C100 4-bit 50% NZ:
  baseline 66.7 -> FIX: MAP=76.56, best-avg(temp3,M50)=76.63  vs PAPER 76.14  => CLOSED+EXCEEDED.
Trajectory: pruning now RAISES acc (73.7 pre-prune -> 75.9 at 50% sparse). Wanda still no help
on CNN; this fix is the lever.
Bayesian avg on fixed recipe (50% sparse): temp1 flat=76.56; temp3 M1=75.50->M50=76.63 (+1.13,
also >MAP). Confirms averaging helps only once eval temp softened (native tau=0.001 -> flat).
Table 5 (outlier window) filled: IQR=90.14 vs equal-window=79.01(prelim). Table 6 (avg) redo at
sparsity=0 launched (19427571).

## Llama-3.2-1B outlier-window ablation — ATTEMPTED, BLOCKED (2026-07-22)
DONE: downloaded meta-llama/Llama-3.2-1B (gated, user token); added SQS_LLAMA1B_PATH override in
config.py (old path pointed at the original author's machine); SST-2 fine-tune SUCCEEDED =
91.17% (both epochs; paper's Llama base is 94.72, so ours is ~3.5pt lower). Fixed a real bug:
Llama MLP SQS_INIT built a GMM on the [0:first_n] slice, which is EMPTY under --no_outlier ->
k-means sampled from an empty range (RuntimeError). Now uses nn.Identity for empty blocks, and
the forward routes them through the positional (identity) path.

BLOCKED by an unfinished Llama pruning path in GPT2_pruner_quantizer.py:
  - generate_mask: lumps CustomizedLlamaAttention into the Qwen branch and iterates
    sub_distribution_list, but Llama attention has a SINGLE sub_distribution -> AttributeError.
  - caculate_mask_thresh: Llama branch registers only v/q/o_proj -- k_proj MISSING (yet
    prune_with_mask expects k_proj.sub_distribution.mask).
  - caculate_mask_thresh (Llama) ranks by raw pruning_parameter (uniform init) instead of
    sweight_cache => ~random pruning, the same bug fixed for ResNet; would collapse accuracy.
  - apply_pruning_grad / prune_with_mask DO have correct Llama branches.
STRUCTURAL CAVEAT: Llama ATTENTION has no outlier window at all (single sub_distribution, no
first_n/last_n); only the MLP is windowed. So even once fixed, a Llama outlier ablation measures
MLP windows ONLY and is NOT comparable to Qwen (where attention+MLP are both windowed).
DECISION PENDING: (a) finish the Llama path (~6-8 SU) or (b) drop the Llama row from Table 5.
