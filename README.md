# VAE Anomaly Detection — Credit Card Fraud

A variational autoencoder trained **only on legitimate transactions**, used as a density model to flag fraud. Nothing fraudulent is ever seen during training; fraud is held out entirely and used only to measure how well the model's likelihood score separates it.

The idea: a VAE squeezed through a narrow latent bottleneck learns the manifold that normal transactions live on. A transaction that doesn't lie on that manifold reconstructs badly and gets a low likelihood — so `-log p(x)` becomes an anomaly score. No fraud labels needed to train, which matters when fraud is 0.17% of your data and labels arrive late.

## Results

Evaluated on the held-out test set: **42,648 legitimate + 492 fraudulent** transactions.

| Model | AUC | AP |
|---|---|---|
| **VAE, `latent=24`** (tuned) | 0.9531 | **0.8066** |
| VAE, `latent=12` (default) | 0.9605 | 0.7384 |
| Isolation Forest (`max_samples=16384`) | 0.9599 | 0.6272 |

**Average precision is the headline metric here, not AUC.** At 0.17% prevalence, ROC-AUC normalises false positives over 284,315 negatives, which flattens every meaningful difference — note that the tuned VAE has the *lowest* AUC of the three while being clearly the best detector. See [finding 3](#3-report-ap-not-auc).

The Isolation Forest baseline is fit on the same held-out legitimate rows the VAE calibrates its threshold on, so the comparison is like-for-like.

## Dataset

The [ULB credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European card transactions from September 2013, of which 492 are fraud (0.1727%). Features `V1`–`V28` are already PCA components (the original fields are confidential); `Time` and `Amount` are raw.

The data is **not committed** — `creditcard.csv` is 143.8 MB, over GitHub's 100 MB per-file hard limit. The notebook's dataset cell resolves it for you, trying in order:

1. an existing `creditcard.csv` under `data/`, matched case-insensitively
2. any `.zip` in `data/`, extracted
3. a Kaggle API download, which needs a `kaggle.json` token

So a local checkout with the file already in `data/` just works, and a fresh Colab runtime fetches it.

Preprocessing: drop `Time`, apply `log1p` to `Amount` (raw skew ≈ 16.98), leaving **29 features**. Legitimate rows are shuffled and split 70/15/15 into train / validation / test. All 492 fraud rows go to test. Standardisation uses the training split's mean and standard deviation only.

## Quick start

**Locally:**

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

Put `creditcard.csv` (or the Kaggle zip) in `data/` and run [VAE.ipynb](VAE.ipynb) top to bottom.

**On Colab:** open [VAE.ipynb](VAE.ipynb) and run it — every dependency is preinstalled. The dataset cell has no local `data/` to find, so it downloads from Kaggle, which needs an API token. Create one at [kaggle.com/settings](https://www.kaggle.com/settings/account) → API → *Create New Token*; the cell prompts you to upload `kaggle.json` once per runtime. To skip that prompt on every reconnect, drop the token in your Drive at `MyDrive/kaggle.json` and mount it — the cell picks it up from there automatically.

Training takes a couple of minutes per run on CPU (~90–150 s per sweep configuration at 25 epochs); CUDA is used automatically when available, so a Colab GPU runtime is noticeably faster for the sweeps.

## How it works

**Model** — a symmetric MLP VAE. Encoder `29 → 128 → 64` with SiLU activations, producing a diagonal Gaussian posterior `(mu, logvar)` over the latent; decoder mirrors it back to 29 outputs. The observation model is Gaussian with a single **learned global scale** `log_var_x`, so the network tunes its own reconstruction precision instead of relying on a hand-set weight between the two loss terms.

`logvar` is clamped to `[-8, 8]`. This isn't there to shape the objective — the KL term is already self-correcting in both directions — it's there to survive the first few hundred steps, when a freshly initialised `fc_logvar` can emit something like 95 and `exp` overflows float32 to `inf`, permanently NaN-ing the run.

**Loss** — Gaussian reconstruction NLL + `beta` × KL to a standard normal prior, with `beta` warming up linearly from 0 to 1 over the first 8 epochs. The warmup lets the decoder learn something useful before the KL term starts pruning latent dimensions, which avoids early posterior collapse.

**Scoring** — anomaly score is `-log p(x)` estimated with an **IWAE bound using L=64 posterior samples**, not the single-sample ELBO. The ELBO is a loose lower bound; the importance-weighted estimate is tighter, which matters when the score itself is the product. Higher score = more anomalous.

**Thresholding** — chosen as a percentile of the *validation* score distribution, which is legitimate-only. That makes the threshold a false-positive budget you pick in advance, with no fraud labels involved.

### Operating points

From the validation-percentile threshold sweep (`latent=12` model):

| Val pct | Threshold | Recall | Precision | F1 | TP | FP |
|---|---|---|---|---|---|---|
| 95.00 | 49.65 | 0.870 | 0.171 | 0.285 | 428 | 2079 |
| 99.00 | 112.10 | 0.815 | 0.510 | 0.627 | 401 | 386 |
| 99.50 | 154.72 | 0.768 | 0.653 | 0.706 | 378 | 201 |
| 99.90 | 317.33 | 0.553 | 0.863 | 0.674 | 272 | 43 |
| 99.95 | 420.43 | 0.411 | 0.918 | 0.567 | 202 | 18 |
| 99.99 | 808.60 | 0.191 | 0.959 | 0.319 | 94 | 4 |

The 99.5th percentile is the F1-optimal point: **77% of fraud caught at 65% precision**, 201 false positives out of 42,648 legitimate transactions.

## What the experiments showed

### 1. `latent` is the hyperparameter that matters

The bottleneck *is* the detector — depth, width and epoch count barely move the result by comparison. The cheapest diagnostic in the notebook is `per_dim_kl`, which counts latent dimensions carrying more than 0.05 nats:

- `latent=12` → **all 12 alive** (1.44–2.32 nats each, total 20.66). Bottleneck not yet binding — search higher.
- `latent=24` → **20 alive, 4 collapsed to exactly 0.0** (total 35.06). Capacity now exceeds what the data needs.

So the legitimate-transaction manifold has an intrinsic dimensionality of **roughly 20**. Read that printout after any run: all dimensions alive means go bigger, several exact zeros means go smaller.

### 2. Validation `-log p(x)` cannot select hyperparameters

Density fit and detection quality are nearly decoupled. Ranked by validation `-logp` (label-free) against final-slice AP:

```
 latent     hidden     val   ap_final
     32  (128, 64)  10.76     0.7673   <- best density fit
     16 (256, 128)  14.04     0.7546
     24  (128, 64)  14.55     0.7945   <- best detection
     16  (128, 64)  17.55     0.7717
     12  (128, 64)  21.40     0.7749
     16   (64, 32)  25.20     0.7732
      8  (128, 64)  26.36     0.7113
      4  (128, 64)  32.34     0.6042   <- worst on both
```

The two criteria agree at the **bottom** — genuine underfitting hurts both — and scramble at the **top**. Validation likelihood detects underfitting but cannot rank configurations that already fit well enough, because it rewards raw capacity for reasons unrelated to catching fraud.

So tuning uses a **40/60 split of the labelled test data**: hyperparameters are chosen on the 40% *selection* slice, and the winner is reported on the disjoint 60% *final* slice the search never touched (fixed mask, seed 20240930, deliberately distinct from `SEED=42`).

### 3. Report AP, not AUC

| Case | AUC | AP |
|---|---|---|
| Isolation Forest, `max_samples=16384` | 0.9599 | 0.6272 |
| VAE `latent=12` | 0.9605 | 0.7384 |
| VAE `latent=24` (tuned) | **0.9531** | **0.8066** |

The tuned VAE has *lower* AUC but much higher AP. Its ROC curve reaches TPR 0.8 at FPR ≈ 0.02 and then climbs slowly — the top 80% of fraud separates cleanly while the last 20% got harder. That is a strictly **better** operating point for an alert-review team, and AUC penalises it, because AUC averages over thresholds nobody would ever deploy.

Use AUC only as a smoke test: below ~0.85 means something is broken.

### 4. Training longer fits the density, not the detector

In a 150-epoch run at `latent=12`, validation `-logp` fell monotonically all the way to epoch 148 (43.34 → 14.81) and never plateaued. Detection, meanwhile, was done early: AP hit 99% of its plateau at **epoch 3**, and across epochs 20–150 it had mean 0.8481, sd 0.0039, and a linear trend of just +0.0020 over 130 epochs — while validation `-logp` improved 32% over that same span.

**40 epochs is roughly 4× more than needed.** Don't use validation likelihood as a stopping signal; 25 epochs is plenty to rank configurations in a sweep.

One caveat found later: at `latent=12` AUC stayed flat with no degradation, but at `latent=24` the AUC curve peaks around epoch 6 and declines steadily through epoch 40. The over-capacity decline is real, but it needs surplus capacity to show up — the 150-epoch experiment used too small a latent to test its own hypothesis.

## Honest caveats

- **The 0.8066 headline is mildly optimistic.** `latent=24` was chosen on a selection slice drawn from the test set, then re-evaluated on the *full* test set, which includes that slice. The leak-free number is the final-slice AP of **0.7945**. Both are reported above; the 0.7945 figure is the one to trust.
- **`latent=24` is not reliably better than its neighbours.** A follow-up fine sweep over `latent` 18–22 with 12 and 24 as anchors, 3 seeds each, gave a pooled across-seed AP standard deviation of **0.0189** — meaning gaps smaller than ~0.038 are not evidence. In that sweep the `latent=24` anchor averaged final AP 0.7662 and `latent=22` came out nominally best at 0.7775 ± 0.0303. Everything in the 18–24 band is statistically indistinguishable, so the single-seed 0.7945 was somewhat lucky. Treat "around 20" as the finding, not any specific value.
- **The per-epoch `AUC*` column peeks at test labels.** It exists as a training diagnostic only, and finding 3 explains why it is the wrong quantity to watch — it should be switched to AP.
- **Findings 4 and the Isolation Forest default comparison are not reproducible from this notebook.** The 150-epoch run and the `max_samples=256` sklearn-default baseline (AUC 0.9521, AP 0.4514 — AP collapses because 256 of 42,647 rows per tree badly underfits 29 dimensions) were run during development but their cells were not retained. Everything else in this README comes from saved output in [VAE.ipynb](VAE.ipynb).

## Repo layout

| Path | What it is |
|---|---|
| [VAE.ipynb](VAE.ipynb) | **The complete working notebook** — data, model, training, 4 figures, threshold sweep, Isolation Forest baseline, coarse and fine hyperparameter sweeps. All results above come from here. |
| [VAE_AnamolyDetection.ipynb](VAE_AnamolyDetection.ipynb) | An in-progress cleaned-up rewrite. Currently 11 cells, ending mid-training-loop — no results yet. |
| `data/` | Dataset, gitignored. See [Dataset](#dataset). |

## Reference

Kingma & Welling, [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114) (2013) — the VAE and the reparameterisation trick.
Burda et al., [*Importance Weighted Autoencoders*](https://arxiv.org/abs/1509.00519) (2015) — the IWAE bound used for scoring.
