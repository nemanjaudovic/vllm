#!/usr/bin/env python3
"""Dump real bf16 weight matrices from a HF safetensors checkpoint as raw files for wvsplitk_accuracy.

  python3 dump_weights.py <hf-repo-or-dir> <regex> [<regex> ...] [--out weights/]
Prints the --weights argument (file:M:K:name,...). Example (Qwen3.5-9B, layer 3 + lm_head):
  python3 dump_weights.py Qwen/Qwen3.5-9B 'layers\\.3\\.(mlp|linear_attn|self_attn)\\..*weight$' 'lm_head.weight'
"""
import argparse, json, os, re
from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("patterns", nargs="+"); ap.add_argument("--out", default="weights")
a = ap.parse_args()
d = a.model if os.path.isdir(a.model) else snapshot_download(a.model, allow_patterns=["*.json", "*.safetensors"])
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
os.makedirs(a.out, exist_ok=True)
args = []
for name, shard in sorted(idx.items()):
    if not any(re.search(p, name) for p in a.patterns):
        continue
    with safe_open(os.path.join(d, shard), "pt") as f:
        t = f.get_tensor(name)
    if t.dim() != 2 or t.dtype != torch.bfloat16 or t.shape[1] % 8:
        continue
    p = os.path.join(a.out, name.replace("/", "_") + ".bin")
    t.contiguous().view(torch.int16).numpy().tofile(p)
    short = ".".join(name.split(".")[-3:-1]) if "layers" in name else name.split(".")[0]  # mlp.down_proj
    args.append(f"{p}:{t.shape[0]}:{t.shape[1]}:{short}")
    print(f"  {name:<70} {tuple(t.shape)}")
print("--weights " + ",".join(args))
