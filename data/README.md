# Input formats

All files are UTF-8. Ids are strings and must be unique within the harmful and the benign half.

## `parallel_corpus.json` (anchor)

One bare request per intent, in each register. `axis.anchor` reads `modern_zh` and `classical_zh`;
other fields are ignored.

```json
{
  "harmful": [
    {"id": "h0001", "en": "...", "modern_zh": "...", "classical_zh": "..."}
  ],
  "benign": [
    {"id": "b0001", "en": "...", "modern_zh": "...", "classical_zh": "..."}
  ]
}
```

## `split.json` (anchor)

Ids of every intent used for evaluation or checkpoint selection. They are removed before the anchor
is computed, so `r̂` and `u` come from training intents only. Other keys are ignored.

```json
{"eval_harm_ids": ["h0001", "..."], "eval_benign_ids": ["b0001", "..."]}
```

## `rot_train.json` (rotation term)

Bare Classical-Chinese requests of the training intents. At every micro-step the trainer samples
`K` from each list and compares their mean last-token states at layer `L`.

```json
{"harm_classical": ["...", "..."], "benign_classical": ["...", "..."]}
```

## `pref_train.jsonl` (preference and commitment terms)

One pair per line. `prompt` is a wrapped request; it is formatted with the model's chat template
(thinking disabled where supported). The trainer appends EOS to both completions.

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

For a harmful prompt, `chosen` is a short refusal and `rejected` is a warn-then-answer or direct
answer. For a benign prompt in the same wrapping, `chosen` is a helpful answer and `rejected` is a
refusal. Extra fields are ignored. The paper's width-1 pool has 3,336 pairs over 583 harmful and 598
benign training intents.
