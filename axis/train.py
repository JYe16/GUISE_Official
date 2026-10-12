"""Train AXIS: one LoRA adapter on a frozen base model with

    L = L_pref + lambda_rot * L_rot + lambda_cmt * L_cmt

  L_pref  DPO on wrapped preference pairs (completion-only log-probabilities; the reference model is
          the same network with the adapter disabled).
  L_rot   1 - cos(Delta_theta, r_hat). At every micro-step, K bare harmful and K bare benign
          Classical-Chinese requests are encoded by the policy; Delta_theta is the difference of their
          mean last-token states at layer L, and r_hat is the frozen refusal direction from
          `python -m axis.anchor`.
  L_cmt   length-normalised NLL of the chosen completion, with weight omega on its first k tokens and
          on the final EOS token.

    python -m axis.train --model Qwen/Qwen3-1.7B --anchor runs/qwen3-1.7b/anchor.npz \
        --pairs data/pref_train.jsonl --rot data/rot_train.json --out runs/qwen3-1.7b/axis_s0

Defaults are the settings of the paper's main comparison. Ablation arms:
    DPO                    --lambda-rot 0 --lambda-cmt 0
    DPO + rot              --lambda-cmt 0
    random / orth anchor   --anchor-mode random | orth
    rot + cmt, no DPO      --lambda-pref 0
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from axis import modeling

DEV = modeling.DEVICE
DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--anchor", required=True, help="anchor .npz written by axis.anchor")
    ap.add_argument("--pairs", help="preference pairs, JSONL with prompt / chosen / rejected")
    ap.add_argument("--rot", required=True, help="JSON with harm_classical / benign_classical request lists")
    ap.add_argument("--out", required=True, help="output directory (adapter/ + train_log.json)")
    # objective
    ap.add_argument("--lambda-pref", type=float, default=1.0)
    ap.add_argument("--lambda-rot", type=float, default=0.1)
    ap.add_argument("--lambda-cmt", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    ap.add_argument("--rot-k", type=int, default=12, help="requests per class per rotation step (K)")
    ap.add_argument("--cmt-first-k", type=int, default=4, help="up-weighted leading tokens (k)")
    ap.add_argument("--cmt-weight", type=float, default=2.0, help="weight on the first k tokens and EOS (omega)")
    ap.add_argument("--anchor-mode", choices=("real", "random", "orth"), default="real",
                    help="real: r_hat; random: same overlap with u as r_hat, random otherwise; orth: orthogonal to u")
    # optimisation
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1, help="preference pairs per micro-step")
    ap.add_argument("--grad-accum", type=int, default=16, help="micro-steps per optimiser step")
    ap.add_argument("--max-len", type=int, default=1024, help="token budget per prompt + completion")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many micro-steps (0 = full epochs)")
    ap.add_argument("--seed", type=int, default=0)
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default=DEFAULT_TARGETS, help="comma-separated module names")
    ap.add_argument("--log-every", type=int, default=20, help="micro-steps between log lines")
    return ap.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_anchor(path, mode, seed):
    """Target direction for L_rot, plus the anchor metadata recorded in the training log.

    random: r' = c0 * u + sqrt(1 - c0^2) * w with w a random unit vector orthogonal to u and r_hat,
            so the placebo keeps the overlap c0 = u . r_hat of the real anchor with the classical axis.
    orth:   r'' = w with w a random unit vector orthogonal to u (zero overlap).
    """
    a = np.load(path)
    layer, r_hat, u = int(a["layer"]), a["r_hat"], a["u"]
    cos0 = float(u @ r_hat)
    if mode == "random":
        e2 = r_hat - cos0 * u; e2 = e2 / np.linalg.norm(e2)
        g = np.random.default_rng(9000 + seed)
        w = g.standard_normal(u.shape)
        w -= (w @ u) * u
        w -= (w @ e2) * e2
        w /= np.linalg.norm(w)
        target = cos0 * u + float(np.sqrt(max(0.0, 1.0 - cos0 ** 2))) * w
    elif mode == "orth":
        g = np.random.default_rng(9100 + seed)
        w = g.standard_normal(u.shape)
        w -= (w @ u) * u
        w /= np.linalg.norm(w)
        target = w
    else:
        target = r_hat
    print(f"anchor {mode}: layer {layer} | cos(u, target) {float(u @ target):.4f} | "
          f"cos(target, r_hat) {float(target @ r_hat):.4f}", flush=True)
    meta = {"anchor_model": str(a["model"]), "anchor_n_harm": int(a["n_harm"]),
            "anchor_n_benign": int(a["n_benign"]), "cos_u_rhat": cos0}
    return layer, target, meta


def pair_ids(tok, prompt, completion, max_len):
    """Token ids and completion mask for one (prompt, completion); the completion ends in EOS."""
    p = tok(modeling.build_inputs(tok, [prompt])[0], add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p + c)[:max_len]
    mask = ([0] * len(p) + [1] * len(c))[:max_len]
    return ids, mask


def pad_batch(seqs, masks, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    cmask = torch.zeros((len(seqs), T), dtype=torch.float)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = torch.tensor(s); attn[i, :len(s)] = 1; cmask[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids.to(DEV), attn.to(DEV), cmask.to(DEV)


def token_logp(model, ids, attn):
    """log p(x_t | x_<t), shape [B, T-1]. Gathers the target logit and subtracts logsumexp, so the
    full-vocabulary log-softmax is never materialised."""
    out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    logits = out.logits[:, :-1, :]
    tgt = ids[:, 1:]
    tk = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits, dim=-1)
    del out, logits
    return (tk - lse).float()


def seq_logp(model, ids, attn, cmask):
    """Summed log-probability of the completion tokens of each sequence."""
    return (token_logp(model, ids, attn) * cmask[:, 1:]).sum(-1)


def main():
    args = parse_args()
    set_seed(args.seed)
    t0 = time.time()
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    HL, target, anchor_meta = build_anchor(args.anchor, args.anchor_mode, args.seed)
    r_mod = torch.tensor(target, dtype=torch.float32, device=DEV)

    # ---- data ----
    if args.lambda_pref > 0 or args.lambda_cmt > 0:
        if not args.pairs:
            raise SystemExit("--pairs is required when --lambda-pref or --lambda-cmt is positive")
        pref = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    else:
        # rotation only: the loop length must be given explicitly
        if not args.max_steps:
            raise SystemExit("--lambda-pref 0 with --lambda-cmt 0 requires --max-steps")
        pref = [None] * args.max_steps
    rot = json.load(open(args.rot, encoding="utf-8"))
    rot_h, rot_b = rot["harm_classical"], rot["benign_classical"]
    print(f"pairs {len(pref)} | rotation requests: harmful {len(rot_h)} benign {len(rot_b)}", flush=True)

    # ---- model + LoRA ----
    tok, model = modeling.load(args.model)
    cap_layer = modeling.get_layers(model)[HL - 1]   # hidden_states[L] is the output of decoder layer L-1
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=[t for t in args.lora_targets.split(",") if t])
    model = get_peft_model(model, lcfg)
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()                  # fp32 LoRA parameters for a stable Adam
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    model.config.use_cache = False
    trunk = model.get_base_model().model             # decoder trunk without the LM head
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    captured = {"active": False, "h": None}
    def hook(m, i, o):
        if captured["active"]:
            captured["h"] = o[0] if isinstance(o, tuple) else o
    handle = cap_layer.register_forward_hook(hook)

    def rotation_loss():
        hi = random.sample(range(len(rot_h)), min(args.rot_k, len(rot_h)))
        bi = random.sample(range(len(rot_b)), min(args.rot_k, len(rot_b)))
        texts = modeling.build_inputs(tok, [rot_h[i] for i in hi] + [rot_b[i] for i in bi])
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256,
                  add_special_tokens=False).to(DEV)
        captured["active"] = True
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                trunk(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
        finally:
            captured["active"] = False
        h = captured["h"][:, -1, :].float()          # last token (left padding)
        d = h[:len(hi)].mean(0) - h[len(hi):].mean(0)
        cos = F.cosine_similarity(d.unsqueeze(0), r_mod.unsqueeze(0)).squeeze()
        return 1.0 - cos, float(cos.detach())

    # ---- train ----
    rot_skipped = 0
    order = list(range(len(pref)))
    log = []
    step = 0; nopt = 0; stop = False
    for ep in range(args.epochs):
        random.shuffle(order)
        for bstart in range(0, len(order) - args.batch_size + 1, args.batch_size):
            l_pref = torch.tensor(0.0, device=DEV); acc = float("nan"); l_cmt = float("nan")
            if args.lambda_pref > 0 or args.lambda_cmt > 0:
                seqs, masks = [], []
                for j in order[bstart:bstart + args.batch_size]:
                    for key in ("chosen", "rejected"):
                        ii, mm = pair_ids(tok, pref[j]["prompt"], pref[j][key], args.max_len)
                        seqs.append(ii); masks.append(mm)
                ids, attn, cmask = pad_batch(seqs, masks, tok.pad_token_id)   # rows: chosen, rejected, ...

                if args.lambda_pref > 0:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pol = seq_logp(model, ids, attn, cmask)
                        with torch.no_grad(), model.disable_adapter():
                            ref = seq_logp(model, ids, attn, cmask)
                    pol_c, pol_r = pol[0::2], pol[1::2]
                    ref_c, ref_r = ref[0::2], ref[1::2]
                    l_pref = -F.logsigmoid(args.beta * ((pol_c - pol_r) - (ref_c - ref_r))).mean()
                    if not torch.isfinite(l_pref):
                        raise SystemExit(f"non-finite preference loss at micro-step {step}")
                    acc = float(((pol_c - pol_r) > (ref_c - ref_r)).float().mean())
                    # backward per term, so that the graphs of the three forwards never coexist
                    ((args.lambda_pref * l_pref) / args.grad_accum).backward()

                if args.lambda_cmt > 0:
                    c_ids, c_attn, c_cmask = ids[0::2], attn[0::2], cmask[0::2]   # chosen rows
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lp = token_logp(model, c_ids, c_attn)
                    cm = c_cmask[:, 1:]; w = cm.clone()
                    for bi in range(cm.shape[0]):
                        nz = cm[bi].nonzero(as_tuple=True)[0]
                        if len(nz) == 0:
                            continue
                        w[bi, nz] = 1.0
                        w[bi, nz[:args.cmt_first_k]] = args.cmt_weight
                        w[bi, nz[-1]] = args.cmt_weight                           # final EOS
                    cmt = -(lp * w).sum() / w.sum().clamp(min=1.0)
                    ((args.lambda_cmt * cmt) / args.grad_accum).backward()
                    l_cmt = float(cmt.detach())

            l_rot = torch.tensor(0.0, device=DEV); cosv = float("nan")
            if args.lambda_rot > 0:
                l_rot, cosv = rotation_loss()
                grads = torch.autograd.grad((args.lambda_rot * l_rot) / args.grad_accum, params,
                                            retain_graph=False, allow_unused=True)
                # a rotation minibatch whose gradient is non-finite is dropped; the other terms stand
                if all((g is None) or bool(torch.isfinite(g).all()) for g in grads):
                    for p, g in zip(params, grads):
                        if g is not None:
                            p.grad = g if p.grad is None else p.grad + g
                else:
                    rot_skipped += 1

            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
                nopt += 1
            if step % args.log_every == 0:
                lp, lr = float(l_pref.detach()), float(l_rot.detach())
                log.append(dict(ep=ep, step=step, l_pref=lp, l_cmt=l_cmt, l_rot=lr,
                                cos_to_anchor=cosv, dpo_acc=acc))
                print(f"ep{ep} step{step:5d} | L_pref {lp:.3f} L_cmt {l_cmt:.3f} L_rot {lr:.3f} "
                      f"| cos(Delta, anchor) {cosv:.3f} | dpo_acc {acc:.2f} "
                      f"| {time.time() - t0:.0f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                stop = True; break
        if stop:
            break
    handle.remove()

    model.save_pretrained(out_dir / "adapter")
    cfg = {k: v for k, v in vars(args).items()}
    json.dump({"config": cfg, "layer": HL, **anchor_meta, "n_pairs": len(pref), "micro_steps": step,
               "optimizer_steps": nopt, "rotation_steps_skipped": rot_skipped, "log": log},
              open(out_dir / "train_log.json", "w"), ensure_ascii=False, indent=1)
    print(f"saved adapter -> {out_dir / 'adapter'} | {nopt} optimizer steps | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
