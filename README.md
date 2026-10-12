# GUISE & AXIS

Official implementation for **How Narrative Wrapping Affects LLM Refusal: A Cross-Language Benchmark and Defense**.

[Paper (arXiv 2610.11005)](https://arxiv.org/abs/2610.11005) · [Project page](https://jye16.github.io/guise/)

Safety-aligned language models refuse a harmful request stated directly but answer the same request
inside a role-play or narrative wrapper. On Qwen3-1.7B, wrapped attacks succeed 89.4% of the time in
English, 93.0% in modern Chinese and 95.7% in Classical Chinese. In the base model, wrapping turns the
harmful–benign direction of the hidden states 77–86° away from the model's refusal direction.

**AXIS** trains one LoRA adapter on a frozen model with

```
L = L_pref + λ_rot · L_rot + λ_cmt · L_cmt
```

| Term | What it does |
|---|---|
| `L_pref` | DPO on wrapped pairs: refuse the harmful prompt, answer the benign prompt in the same wrapping |
| `L_rot`  | `1 − cos(Δθ, r̂)`: pulls the policy's Classical-Chinese harm direction onto the refusal direction `r̂` of bare modern-Chinese requests |
| `L_cmt`  | Length-normalised NLL of the chosen completion, up-weighting its first `k` tokens and its EOS, so a refusal is carried to the end instead of turning into warn-then-answer |

This repository contains the AXIS training pipeline: anchor estimation on the frozen base model,
then the trainer. The trainer is the script behind the paper's numbers, cleaned up; on the same inputs
it produces bit-identical adapters.

| | |
|---|---|
| **Models** | Qwen3-1.7B · Qwen3-4B · GLM-4-9B |
| **Adapter** | LoRA `r = 16`, `α = 32` |
| **Weights** | `λ_rot = 0.1` · `λ_cmt = 0.2` · `β = 0.1` |
| **Budget** | 2 epochs, 417 updates, lr `2e-5` |
| **Hardware** | one A10G, about 75 minutes for Qwen3-1.7B |
| **Data** | GUISE release coming soon |

## Installation

```bash
conda create -n guise python=3.11 -y && conda activate guise
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # pick the CUDA build for your GPU
pip install -r requirements.txt
```

## Data

Training reads four files. Their formats are in [`data/README.md`](data/README.md).

| File | Used by | Content |
|---|---|---|
| `parallel_corpus.json` | `axis.anchor` | bare requests per intent in modern Chinese and Classical Chinese |
| `split.json` | `axis.anchor` | ids of the evaluation intents, excluded from the anchor |
| `rot_train.json` | `axis.train` | bare Classical-Chinese requests of the training intents, harmful and benign |
| `pref_train.jsonl` | `axis.train` | preference pairs `(prompt, chosen, rejected)` |

## Training: two commands per model

**1. Anchor.** Encode the corpus with the frozen base model and estimate the refusal direction `r̂`
(bare modern Chinese) and the classical harm direction `u` at layer `L`, on training intents only.

```bash
python -m axis.anchor --model Qwen/Qwen3-1.7B \
    --corpus data/parallel_corpus.json --split data/split.json \
    --layer 24 --out runs/qwen3-1.7b/anchor.npz
```

**2. Adapter.** Defaults are the settings of the paper's main comparison
(`λ_rot = 0.1`, `λ_cmt = 0.2`, `K = 12`, `k = 4`, `ω = 2`, `β = 0.1`, LoRA `r = 16`, `α = 32`,
lr `2e-5`, two epochs, effective batch 16, 1,024-token budget).

```bash
python -m axis.train --model Qwen/Qwen3-1.7B --anchor runs/qwen3-1.7b/anchor.npz \
    --pairs data/pref_train.jsonl --rot data/rot_train.json \
    --out runs/qwen3-1.7b/axis_s0 --seed 0
```

`scripts/train_paper.sh` runs both steps with the per-model settings below:

```bash
bash scripts/train_paper.sh qwen3-1.7b 0                                      # model, seed
TAG=random_s0 bash scripts/train_paper.sh qwen3-1.7b 0 --anchor-mode random    # extra flags go to axis.train
```

| Model | Layer `L` | LoRA target modules | Hardware in the paper |
|---|---|---|---|
| `Qwen/Qwen3-1.7B` | 24 | `q,k,v,o,gate,up,down_proj` | one A10G, about 75 min |
| `Qwen/Qwen3-4B` | 19 | `q,k,v,o,gate,up,down_proj` | H100 |
| `zai-org/GLM-4-9B-0414` | 21 | `q,k,v,o,gate_up,down_proj` | H100 |

`--layer auto` selects the layer where the modern harmful/benign contrast is most separable.

### Ablation arms

| Arm | Flags added to `axis.train` |
|---|---|
| DPO | `--lambda-rot 0 --lambda-cmt 0` |
| DPO + rot | `--lambda-cmt 0` |
| AXIS, random anchor | `--anchor-mode random` |
| AXIS, orthogonal anchor | `--anchor-mode orth` |
| rot + cmt, no DPO | `--lambda-pref 0` |

The paper's ablation and mechanism experiments use `--lambda-rot 1`.

### Outputs

`<out>/adapter/` is a PEFT LoRA adapter. `<out>/train_log.json` holds the full configuration, the
anchor metadata and the loss trace (`L_pref`, `L_cmt`, `L_rot`, cosine between the policy's
classical harm direction and the anchor).

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda")
model = PeftModel.from_pretrained(base, "runs/qwen3-1.7b/axis_s0/adapter")
```

## Layout

```
axis/modeling.py     model loading, chat template, decoder-layer lookup
axis/anchor.py       refusal direction r̂ and classical harm direction u from the base model
axis/train.py        AXIS trainer (DPO + rotation + commitment, one LoRA adapter)
scripts/train_paper.sh
data/README.md       input file formats
```

Bit-identical here covers the adapters and loss traces of all five arms in the ablation table.

## Citation

```bibtex
@misc{ye2026narrativewrappingaffectsllm,
      title={How Narrative Wrapping Affects LLM Refusal: A Cross-Language Benchmark and Defense}, 
      author={Zhankai Ye and Yanning Wang and Yukai Jin and Bo Mei and Fangyi Li and Wei Wang and Shangqian Gao and Xin Liu},
      year={2026},
      eprint={2610.11005},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.11005}, 
}
```

## License

MIT. See [`LICENSE`](LICENSE).
