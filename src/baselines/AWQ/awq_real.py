"""Faithful AWQ (Lin et al., MLSys 2024) for the SST-2 Qwen2.5-0.5B classifier.

The repo's original quantization.py is a STUB (it evaluates at full precision). This is
the real algorithm, weight-only, as fake-quantization (quantize->dequantize to fp16 so no
custom CUDA kernels are needed -- accuracy is identical to a real AWQ kernel, which only
changes speed/memory). This is exactly how the AWQ paper reports pseudo-quantized accuracy.

AWQ per Linear W [out,in] with input activations X [.,in]:
  1. per-input-channel activation scale  a_j = mean_t |X[t,j]|
  2. search s = a^alpha  (alpha in a grid) that best preserves the layer output:
       W X = (W diag(s)) (diag(1/s) X)  -> quantize the better-conditioned  W diag(s),
       then fold 1/s back:  W_awq = Q(W diag(s)) diag(1/s).  Salient (high-activation)
       channels are scaled up before rounding, shrinking their relative quant error.
  3. Q(.) = group-wise (g=128) asymmetric INT-w_bit round-to-nearest.
  4. pick alpha minimizing || W_awq X - W X ||  (mean over calib tokens).

Reports SST-2 accuracy for each (w_bit, group) and the honest effective bits/weight, so
it can be plotted against SQS on the same axis (see analysis/effective_bits.py).

Usage (GPU job, sqs env, offline):
  python awq_real.py --model_path $SQS_QWEN05B_PATH --w_bit 4 --group 128 --n_calib 128
"""
import argparse, os, sys, math
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torchmetrics.classification import MulticlassAccuracy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../SQS"))
from SQS.config import model_config  # reuse the same model registry

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def group_quant_dequant(W, w_bit, group):
    """Asymmetric per-group RTN over the INPUT dim (columns). W: [out, in]. Returns fp W_hat."""
    out, cin = W.shape
    g = group if group and group > 0 else cin
    pad = (g - cin % g) % g
    if pad:
        W = torch.cat([W, W.new_zeros(out, pad)], dim=1)
    Wg = W.reshape(out, -1, g)                      # [out, n_groups, g]
    qmax = 2 ** w_bit - 1
    wmin = Wg.min(dim=2, keepdim=True).values
    wmax = Wg.max(dim=2, keepdim=True).values
    scale = (wmax - wmin).clamp(min=1e-8) / qmax
    zp = torch.round(-wmin / scale)
    q = torch.clamp(torch.round(Wg / scale) + zp, 0, qmax)
    Wg_hat = (q - zp) * scale
    W_hat = Wg_hat.reshape(out, -1)[:, :cin]
    return W_hat


@torch.no_grad()
def awq_layer(W, x_abs_mean, calib_X, w_bit, group, grid=20):
    """Return fake-quantized W_awq using the best AWQ scale. calib_X: [N, in] sample of inputs."""
    best = None
    ref = calib_X @ W.t()                            # [N, out] fp reference output
    a = x_abs_mean.clamp(min=1e-6)
    for i in range(grid + 1):
        alpha = i / grid
        s = a.pow(alpha)                             # [in]
        s = s / (s.mean() + 1e-8)                    # normalize to keep magnitudes sane
        Ws = W * s.unsqueeze(0)                      # scale columns
        Q = group_quant_dequant(Ws, w_bit, group)
        W_awq = Q / s.unsqueeze(0)                   # fold 1/s back
        err = ((calib_X @ W_awq.t()) - ref).pow(2).mean().item()
        if best is None or err < best[0]:
            best = (err, alpha, W_awq)
    return best[1], best[2]


def _load_sst2():
    """Offline-safe SST-2: load the cached parquet directly (the 'glue' script isn't
    cached on the offline compute nodes, so load_dataset('glue',...) fails there)."""
    import glob
    base = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                        "hub/datasets--glue/snapshots")
    val = glob.glob(base + "/*/sst2/validation-*.parquet")
    tr = glob.glob(base + "/*/sst2/train-*.parquet")
    if val and tr:
        return load_dataset("parquet", data_files={"validation": val, "train": tr})
    return load_dataset("nyu-mll/glue", "sst2")   # fallback (needs internet)


def build_dataloaders(model_name, task, n_calib, max_len=128):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw = _load_sst2()

    def enc(split, n=None):
        d = raw[split]
        if n:
            d = d.select(range(min(n, len(d))))
        b = tok(d["sentence"], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        return b, torch.tensor(d["label"])
    calib, _ = enc("train", n_calib)
    ev, ev_y = enc("validation")
    return tok, calib, ev, ev_y


@torch.no_grad()
def collect_inputs(model, calib, n_cache=512):
    """Run one calib forward, caching a sample of each proj's input rows + |X| mean."""
    store = {}
    handles = []

    def mk(name):
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            s = store.setdefault(name, {"sum": torch.zeros(x.shape[-1], device=x.device), "cnt": 0, "X": []})
            s["sum"] += x.abs().sum(0); s["cnt"] += x.shape[0]
            if sum(t.shape[0] for t in s["X"]) < n_cache:
                s["X"].append(x[:64].float())
        return hook

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and n.endswith(PROJ) and "score" not in n:
            handles.append(m.register_forward_pre_hook(mk(n)))
    model.eval()
    b = {k: v.to(dev) for k, v in calib.items() if torch.is_tensor(v)}
    model(**{k: v for k, v in b.items() if k != "labels"})
    for h in handles:
        h.remove()
    return {n: {"abs_mean": s["sum"] / s["cnt"], "X": torch.cat(s["X"])[:n_cache]} for n, s in store.items()}


@torch.no_grad()
def evaluate(model, ev, ev_y, num_labels, bs=64):
    model.eval()
    metric = MulticlassAccuracy(num_classes=num_labels, average="micro").to(dev)
    ids, mask = ev["input_ids"], ev["attention_mask"]
    for i in range(0, ids.shape[0], bs):
        out = model(input_ids=ids[i:i+bs].to(dev), attention_mask=mask[i:i+bs].to(dev))
        metric.update(out.logits.argmax(-1), ev_y[i:i+bs].to(dev))
    return metric.compute().item()


def main(a):
    mc = model_config[a.model_name]
    config = AutoConfig.from_pretrained(a.model_name, num_labels=2, finetuning_task=a.task, trust_remote_code=True)
    src = a.model_path or mc["from_pretrained"]
    model = AutoModelForSequenceClassification.from_pretrained(
        src, config=config, attn_implementation=mc["attn_implementation"],
        torch_dtype=torch.float32, trust_remote_code=True).to(dev)
    model.config.pad_token_id = config.eos_token_id

    tok, calib, ev, ev_y = build_dataloaders(a.model_name, a.task, a.n_calib, a.max_len)
    print(f"[AWQ] base (fp32) acc = {evaluate(model, ev, ev_y, 2):.4f}", flush=True)

    stats = collect_inputs(model, calib)
    print(f"[AWQ] calibrated {len(stats)} projections; quantizing w_bit={a.w_bit} group={a.group}", flush=True)

    linears = {n: m for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.endswith(PROJ) and "score" not in n}
    for n, m in linears.items():
        if n not in stats:
            continue
        W = m.weight.data.float()
        alpha, W_awq = awq_layer(W, stats[n]["abs_mean"], stats[n]["X"], a.w_bit, a.group)
        m.weight.data.copy_(W_awq.to(m.weight.dtype))
    acc = evaluate(model, ev, ev_y, 2)

    # effective bits/weight (asymmetric group RTN: w_bit + (scale_fp16 + zp_wbit)/group)
    eff = a.w_bit + (16 + a.w_bit) / a.group
    print(f"[AWQ] RESULT  w_bit={a.w_bit} group={a.group}  acc={acc:.4f}  "
          f"eff_bits={eff:.3f}  compression={32/eff:.2f}x", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--model_path", default=os.environ.get("SQS_QWEN05B_PATH"))
    p.add_argument("--task", default="sst2")
    p.add_argument("--w_bit", type=int, default=4)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--n_calib", type=int, default=128)
    p.add_argument("--max_len", type=int, default=128)
    main(p.parse_args())
