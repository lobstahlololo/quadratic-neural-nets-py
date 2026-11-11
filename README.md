# Quadratic NN test

This repo tests quadratic NNs where instead of $W*x+b$ we have $W_quad*x^2+W_lin+b$.


Tests I made showcase promising results: quadratic FFNs matching linear models with multiple times greater params (for example, on a 120k quadratic transformer I got 2.65 validation loss on TinyStories, compared to 2.39 for the 1M linear transformer), though I was unable to train on the full dataset due to compute constraints.

I'm also testing on BabyLM, where I was able to reach 62% BLiMP on a 5 layer 1M Quadratic *without* dropout, label smoothing, layerscale, or weight decay (and a bug in ROPE!). Trained om BabyLM Strict Small for 6 epochs before I stopped it. Full results will be shown when I optimize hyperparameters

This likely works as quadratic equations are more expressive than linear, while being a superset of them if the quadratic term is set to zero. 

## Quick Start

Run:

```
git clone https://github.com/supastishn/quadratic-neuralnets
cd quadratic-neuralnets/documentation
g++ -DQQ_BLAS_GENERIC -DTRAINING_ON example_mnist.cpp -o mnist
```
Congrats! You've now trained a small scale MNIST digit recognition model using quadratic neurons.

## Proofs and Motivations

1. Taylor Series Math

Linear approximation is $f(x+h) = f(x)+hf'(x)$

Quadratic approximation is $f(x+h) = f(x)+hf'(x) + h^2/2!*f''(x)$

As long as f''(x) isn't 0 the quadratic approximation is better. Also:

Linear error scales with the next term, h²/2!. Whereas quadratic error scales with the next term, h³/3!. So quadratic should be more accurate by a factor of $cuberoot(2!/3!)*h^2/3$ which is approximately 0.693*n_linear^2/3

Note however that is only about expressiveness of single neurons: when we assume a function with many inputs this breaks down as it requires the sum of all pairs. I'm not good enough at math to figure out how much better linear vs squarewise is in terms of error with multiple inputs.

But the point remains: quadratic nodes are more expressive than linears for curves as long as f''(x) exists: a quadratic could represent a curve in a few nodes that takes linear tens of nodes.

2. Simpson's Rule

In integration, Simpson's rule using parabola gives a more accurate approximation of the integral. So I thought the same accuracy should apply here.

Simpson's rule is around sqrt(linear) better, though Simpson's Rule can also integrate cubics due to cool symmetry which is what makes it sqrt() not root[1.5]

Obviously integration is somewhat unrelated but it does suggest that quadratic functions can be more accurate than linear ones. 

3. KANs

Kolmogorov Arnold networks involve polynomials too, except with B-splines. However, KANs are super hard to compute. I thought that if we added polynomials to the mix of standard NN we might get some of the same benefits with just one extra matmul.

4. Linears are a subset of quadratics

Linear equations are a subset of quadratics where the quadratic weight is 0. So in theory, a quadratic model should be able to learn to be linear if that's what the data requires. So it should be *at least* as good as a linear model, with options to use squared if that's what's needed.

## FAQ

### Why not just more linear layers?

While two linear layers with a non-linear function can *approximate* a quadratic, it does not give the same expressiveness as a natively quadratic layer. 

## Challenges

1. Quadratic layers are more expensive to compute, meaning a quadratic model with half the layer  count of a linear model will actually have the same parameters as a linear

2. Quadratics are more expressive but also that means they can oscillate and overfit (runge's phenomenon). Which is exactly why polynomial regression doesn't work. However, Runge's phenomenon mainly applies for high order polynomials not just a quadratic. Just in case I'm planning on careful L-n penalties and dropout which is already standard

3. Since to calculate $dOutput/dInput$ you calculate 2ax+b, not just b, the gradients will be more prone to expllosion and vanishing. So I need to add gradient clipping, dropout, careful initialization, label smoothing, etc.

4. Finally, the backprop is just more complex. So it's harder to write.

## Experiment 001 - TinyStories baseline (verified)

One bounded run of the existing research harness (`python/research`), using the
frozen architecture in `python/model.py`, the existing `Trainer`, optimizer,
scheduler, streaming data pipeline and character tokenizer unchanged. No
hyperparameter was tuned. This is a test of whether the current quadratic
transformer learns anything measurable on TinyStories, not a benchmark.

Numbers below are **measured**; the short interpretation notes say what I read
into them.

### Configuration (**measured**)

| quantity | value |
| --- | --- |
| parameters | 2,229,491 |
| architecture | `Embedding(243,128)` -> 4 blocks of [`QuadNorm` -> 4 sequential single-head attentions -> `QuadNorm` -> `QuadraticLinear(128->512)` -> ReLU -> `QuadraticLinear(512->128)`] -> `QuadNorm` -> `QuadraticLinear(128->243)`; no residual connections, no positional embeddings |
| data | `data/TinyStories/tinystories-train.txt` (1.92 GB) train, `data/TinyStories/tinystories-valid.txt` (19.4 MB) validation |
| tokenizer | existing character-level tokenizer, built from the train split, **243 tokens** |
| context / batch / window stride | 64 / 8 / 32 |
| optimizer | AdamW, lr 0.001, min_lr 1e-5, weight decay 0.01, `W_quad` scale 0.7 (`lr_eff = lr/batch_size = 0.000125`) |
| scheduler | existing per-epoch cosine over `steps_per_epoch = 7,510,808` (unchanged) |
| training budget | **500 optimizer steps** = `epochs=1` with `max_steps_per_epoch=500`, i.e. **0.0067% of one epoch** (256,000 tokens); one full epoch was never run |
| validation subset | 256 fixed batches = 131,072 tokens, identical before and after training |
| hardware | CPU only (no CUDA available), PyTorch 2.13.0, Python 3.14.7, Windows |

### Results (**measured**)

| metric | before training | after 500 steps |
| --- | --- | --- |
| validation cross entropy | 5.493916 | 3.198449 |
| validation perplexity | 243.208 | 24.495 |
| training loss | - (fresh init is uniform over the vocab, ~ln 243 = 5.49) | 3.196237 (mean of last 50 steps) / 3.248941 (step 500) |

- validation improvement: **-2.295467 (-41.78%)**, so yes, this run learned
  something measurable that transfers to held-out text.
- training loss fell steadily (with step-to-step noise): 5.386 at step
  10, 4.833 at 100, 4.145 at 200, 3.598 at 300, 3.397 at 400, 3.249 at 500.
  The learning rate stayed at ~0.001 (only 0.0067% of the cosine schedule).
- runtime: 202.7 s for the 500 steps plus end-of-epoch validation and the two
  checkpoint writes -> **2.47 steps/s, 1,263 tokens/s** on CPU. A further 338 s
  went to one-time preprocessing (tokenizer build over 1.92 GB + one token-count
  pass) and is excluded from throughput.
- checkpoint `checkpoint_final.pt` (`global_step = 500`) reloads into a fresh
  model and reproduces the final validation loss exactly (3.198449).

### Generation (measured output, interpretation in the notes)

Settings: temperature 0.8, `max_new_tokens=128`, no top-k/top-p, context 64,
5 seeds (1234-1238) plus one greedy sample (temperature 0). Prompts included
`Once upon a time, there was a little girl named Lily.`, `The sun was shining
and the bird said`, and `In a small house near the forest, a cat sat on a mat.`

Example output (temperature 0.8):

```
Once upon a time, there was a little girl named Lily. oredinld n 'htuhwcse−sen  nirah🌹akat td eleeks"dCpla童ow  il rsewao eywo  odmli
  eltorer h edtly  tsils tn yarisreth oowehk cso
```

All 6 samples had finite logits and token IDs inside [0, 243): no NaNs or
invalid indices.

**Interpretation:** the output is currently incoherent character-level text,
not stories and not words. The model has picked up whitespace, punctuation,
letter frequencies and some short local patterns; it has not learned word
structure at this budget. That is what 0.0067% of an epoch buys, and it is why
this experiment says nothing yet about the architecture's language ability.

### Overfitting and repetition

- **Measured:** train/val gap at the end of the run is +0.0022 (train 3.196237
  vs val 3.198449) - no obvious overfitting signal, as expected this early.
- Caveat: only two validation points exist (start and end), so this is not a
  held-out learning curve; it is a check that validation loss did not diverge
  from training loss.
- **Measured:** sampled text shows no n-gram looping (distinct 8-gram ratio 1.0,
  longest repeated substring 3-5 characters).
- **Measured:** greedy decoding (temperature 0) collapses to the single most
  probable token, emitting 128 spaces after the prompt.
- **Interpretation:** the greedy collapse is the usual low-entropy failure of an
  under-trained character model rather than a scheduling or gradient bug; the
  metrics file shows no NaN or unstable gradient norms (quadratic 71.7, linear
  265.2, other 256.5 at step 10, all finite through step 500).
- Samples contain non-ASCII characters (e.g. emoji, CJK) because the corpus
  character vocabulary itself contains them; they are valid tokens, not
  corruption.

### Reproducibility

- seed 1; the model is initialized by the deterministic C++-matching
  `init_like_cpp(seed)`, the data pipeline is unshuffled and deterministic, and
  the trainer writes one metrics row per 10 steps.
- commands run from the repo root (the first does everything, the second only
  rebuilds evaluation, samples and report from the saved checkpoint):

```
python python/research/experiment_001.py
python python/research/experiment_001.py --post-only --run-log runs/experiment_001_run.log
```

- the driver calls `research.trainer.Trainer.run` with the same values the CLI
  exposes as `--epochs 1 --max-steps-per-epoch 500 --max-valid-batches 256`; the
  model, trainer, optimizer, scheduler and data code are unmodified.
- artifacts (git-ignored) in `runs/experiment_001/`:
  `baseline_untrained.pt`, `baseline_eval.json`, `metrics.jsonl` (one row per 10
  steps plus the epoch row), `checkpoint_epoch1.pt`, `checkpoint_final.pt`,
  `final_eval.json`, `generations.json` (all samples and settings),
  `report.json`, `report.md` (human-readable report). Run logs:
  `runs/experiment_001_run.log`, `runs/experiment_001_post.log`.

## Notes

My plan is to make this *exactly* the same as a transformer just with squarewise FFN. So still some Re/GeLu, attention, etc.

My AI tool, RComp, was used for certain parts of the code (mainly catching errors and bugs, as it's quite hard to on a phone, and making the AttentionHook)
