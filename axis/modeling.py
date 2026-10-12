"""Model loading, chat formatting and decoder-layer lookup shared by the anchor and training scripts."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:0"


def load(model_id, dtype=torch.bfloat16):
    """Left-padded tokenizer and the base model on a single GPU, in eval mode."""
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map={"": DEVICE})
    model.eval()
    return tok, model


def get_layers(model):
    """The decoder-layer ModuleList of a causal LM."""
    import torch.nn as nn
    for path in ("model.layers", "model.model.layers", "language_model.model.layers"):
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8 and (best is None or len(mod) > len(best)):
            best = mod
    if best is None:
        raise RuntimeError("decoder layers not found")
    return best


def build_inputs(tok, prompts, system=None):
    """Apply the chat template to single-turn user prompts (thinking disabled where supported)."""
    texts = []
    for p in prompts:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": p})
        try:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(t)
    return texts
