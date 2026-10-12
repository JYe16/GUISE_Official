"""Estimate the refusal direction and the classical harm direction from the frozen base model.

For a set of matched harmful and benign requests, the harm direction at layer L is the difference of
mean last-token hidden states, harmful minus benign. The refusal direction r_hat is the unit harm
direction of bare modern-Chinese requests (Arditi et al., 2024); u is the unit harm direction of the
bare Classical-Chinese renderings of the same intents. Both are computed on training intents only:
every id listed under `eval_harm_ids` / `eval_benign_ids` in the split file is dropped first.

    python -m axis.anchor --model Qwen/Qwen3-1.7B --corpus data/parallel_corpus.json \
        --split data/split.json --layer 24 --out runs/qwen3-1.7b/anchor.npz

`--layer auto` picks the layer whose modern harmful/benign contrast is most separable (ROC AUC of
the projection on the mean-difference direction, training intents only).
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from axis import modeling


@torch.no_grad()
def last_token_states(tok, model, prompts, layers, batch_size):
    """Hidden states at the last prompt token for the given hidden_states indices: [n, len(layers), H].

    The whole corpus is encoded in file order, so batch composition (and with it the left-padding
    length) does not depend on which intents are later kept for the anchor.
    """
    texts = modeling.build_inputs(tok, prompts)
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(modeling.DEVICE)
        hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states, dim=1)  # [B, L+1, T, H]
        out.append(hs[:, :, -1, :].float().cpu()[:, layers])
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r", flush=True)
    print(flush=True)
    return torch.cat(out).numpy()


def unit(v):
    return v / np.linalg.norm(v)


def auc(pos, neg):
    """ROC AUC with average ranks for ties."""
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--corpus", required=True, help="parallel corpus JSON (see data/README.md)")
    ap.add_argument("--split", required=True, help="split JSON with eval_harm_ids / eval_benign_ids")
    ap.add_argument("--layer", default="auto",
                    help="hidden_states index L (0 = embeddings), or 'auto'")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True, help="output .npz")
    args = ap.parse_args()

    corpus = json.load(open(args.corpus, encoding="utf-8"))
    harm, benign = corpus["harmful"], corpus["benign"]
    split = json.load(open(args.split, encoding="utf-8"))
    ev_h, ev_b = set(split["eval_harm_ids"]), set(split["eval_benign_ids"])
    keep_h = [i for i, x in enumerate(harm) if x["id"] not in ev_h]
    keep_b = [i for i, x in enumerate(benign) if x["id"] not in ev_b]
    if not keep_h or not keep_b:
        raise SystemExit("no training intents left after removing the evaluation ids")
    print(f"corpus: harmful {len(harm)} benign {len(benign)} -> anchor intents "
          f"{len(keep_h)} / {len(keep_b)}", flush=True)

    tok, model = modeling.load(args.model)
    n_states = len(modeling.get_layers(model)) + 1
    # hidden_states[0] is the embedding output: the last prompt token is the same chat-template
    # token for every request, so it carries no contrast and is never a candidate.
    layers = list(range(1, n_states)) if args.layer == "auto" else [int(args.layer)]
    if not all(1 <= l < n_states for l in layers):
        raise SystemExit(f"--layer must be in [1, {n_states - 1}]")

    acts = {}
    for cls, items in (("harm", harm), ("benign", benign)):
        for reg, key in (("modern", "modern_zh"), ("classical", "classical_zh")):
            print(f"encode {cls}/{reg} (n={len(items)})", flush=True)
            acts[f"{cls}_{reg}"] = last_token_states(tok, model, [x[key] for x in items],
                                                     layers, args.batch_size)

    if args.layer == "auto":
        def separability(j):
            hm, bm = acts["harm_modern"][:, j, :][keep_h], acts["benign_modern"][:, j, :][keep_b]
            d = hm.mean(0) - bm.mean(0)
            if not np.linalg.norm(d) > 0:
                return 0.0
            d = unit(d)
            return auc(hm @ d, bm @ d)
        j = int(max(range(len(layers)), key=separability))
    else:
        j = 0
    L = layers[j]

    hm, bm = acts["harm_modern"][:, j, :][keep_h], acts["benign_modern"][:, j, :][keep_b]
    hc, bc = acts["harm_classical"][:, j, :][keep_h], acts["benign_classical"][:, j, :][keep_b]
    r_hat = unit(hm.mean(0) - bm.mean(0))
    u = unit(hc.mean(0) - bc.mean(0))
    cos0 = float(u @ r_hat)
    angle = float(np.degrees(np.arccos(np.clip(cos0, -1.0, 1.0))))
    print(f"layer {L}: angle(classical harm direction, refusal direction) = {angle:.1f} deg "
          f"(cos {cos0:.4f})", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, layer=np.int64(L), r_hat=r_hat, u=u, model=np.str_(args.model),
             n_harm=np.int64(len(keep_h)), n_benign=np.int64(len(keep_b)))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
