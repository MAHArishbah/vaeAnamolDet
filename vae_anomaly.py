#!/usr/bin/env python
"""VAE anomaly detection on the ULB credit-card fraud dataset.

A variational autoencoder is trained on legitimate transactions only; the IWAE
estimate of -log p(x) then serves as an anomaly score. Fraud is never seen in
training and is used purely for evaluation.

Ported from VAE.ipynb. The maths is unchanged, so the numbers in README.md
reproduce; the training loop is reorganised to keep the GPU busy (see
train_vae). Figures are written to --figdir instead of being shown, since
plt.show() blocks in a script.

Subcommands
-----------
    train        train one configuration, evaluate, write figures
    baseline     the same, plus an Isolation Forest comparison
    sweep        coarse grid over latent size and hidden widths
    fine-sweep   latent 18-22 with 12/24 anchors, multiple seeds

Examples
--------
    python vae_anomaly.py train
    python vae_anomaly.py train --latent 24 --epochs 40 --batch 8192
    python vae_anomaly.py baseline
    python vae_anomaly.py sweep --epochs 25
    python vae_anomaly.py fine-sweep --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, TypedDict

import matplotlib
matplotlib.use("Agg")                 # no display in a script; must precede pyplot

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

LOG2PI = math.log(2 * math.pi)
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"

SEED = 42
DEVICE = "cpu"                        # set by main() from --device

# --- validated chart tokens (light mode) --------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXLINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
C_LEGIT, C_FRAUD, C_THR = "#2a78d6", "#eb6834", "#d03b3b"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 120,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXLINE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "lines.linewidth": 2.0,
    "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 10,
})


# ============================================================================
# dataset resolution
# ============================================================================

def _find_csv(d: Path):
    """Match creditcard.csv case-insensitively. Kaggle ships it lowercase, but a
    Windows copy may be capitalised -- Windows forgives that and Linux does not."""
    if not d.exists():
        return None
    return next((p for p in d.rglob("*")
                 if p.is_file() and p.name.lower() == "creditcard.csv"), None)


def _unzip_local(d: Path):
    for z in sorted(d.glob("*.zip")):
        print(f"extracting {z.name}")
        with zipfile.ZipFile(z) as archive:
            archive.extractall(d)
    return _find_csv(d)


def _install_kaggle_token() -> bool:
    """Put kaggle.json where the CLI looks for it. False if we have none."""
    target = Path.home() / ".kaggle" / "kaggle.json"
    if target.exists():
        return True
    for cand in (Path.cwd() / "kaggle.json",
                 Path("/content/kaggle.json"),
                 Path("/content/drive/MyDrive/kaggle.json")):
        if cand.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(cand.read_bytes())
            target.chmod(0o600)
            print(f"using token {cand}")
            return True
    return False


def resolve_data(data_dir: Path) -> Path:
    """Existing CSV -> local zip -> Kaggle download. Returns the CSV path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv = _find_csv(data_dir) or _unzip_local(data_dir)
    if csv is not None:
        return csv

    if not _install_kaggle_token():
        raise FileNotFoundError(
            f"no creditcard.csv under {data_dir}, and no kaggle.json token.\n"
            f"Either drop the CSV in that folder yourself, or create a token at\n"
            f"  https://www.kaggle.com/settings/account  ->  API  ->  Create New Token")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    print(f"downloading {KAGGLE_DATASET} -> {data_dir}")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
                    "-p", str(data_dir), "--unzip"], check=True)

    csv = _find_csv(data_dir)
    if csv is None:
        raise FileNotFoundError(f"download finished but no creditcard.csv under {data_dir}")
    return csv


def make_data(csv_path: Path):
    """Legit rows (shuffled) and fraud rows. Fraud never enters training."""
    df = pd.read_csv(csv_path)
    y = df.pop("Class").to_numpy().astype(int)

    df["Amount"] = np.log1p(df["Amount"])     # heavy right tail -> Gaussian NLL
    df = df.drop(columns=["Time"])            # global clock, not a feature

    X = df.to_numpy(dtype=np.float32)
    rng = np.random.default_rng(SEED)
    X_legit = X[y == 0]
    X_legit = X_legit[rng.permutation(len(X_legit))]    # file is time-ordered
    return X_legit, X[y == 1], list(df.columns)


# ============================================================================
# model
# ============================================================================

class VAE(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), latent=12):
        super().__init__()

        enc, prev = [], in_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        self.enc = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent)
        self.fc_logvar = nn.Linear(prev, latent)

        dec, prev = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        self.dec = nn.Sequential(*dec)
        self.fc_out = nn.Linear(prev, in_dim)

        # learned global observation noise: log of sigma_x^2. This auto-balances
        # reconstruction against KL, so beta stays at 1.
        self.log_var_x = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        h = self.enc(x)
        # the clamp is not there to shape the objective -- the KL is already
        # self-correcting in both directions. It is there to survive the first
        # few hundred steps, when a fresh fc_logvar can emit ~95 and exp()
        # overflows float32 to inf, NaN-ing the run permanently.
        return self.fc_mu(h), self.fc_logvar(h).clamp(-8.0, 8.0)

    def decode(self, z):
        return self.fc_out(self.dec(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)    # reparam
        return self.decode(z), mu, logvar


def elbo_loss(x, x_hat, mu, logvar, log_var_x, beta=1.0):
    """Returns (loss, nll, kl) as tensors.

    The notebook returned nll/kl already .item()'d, which forced a host-device
    sync on every step. Keeping them as tensors lets the caller accumulate on
    device and sync once per epoch instead.
    """
    nll = 0.5 * (((x - x_hat) ** 2) / log_var_x.exp()
                 + log_var_x + LOG2PI).sum(1)                      # (B,)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)       # (B,)
    return (nll + beta * kl).mean(), nll.mean().detach(), kl.mean().detach()


@torch.no_grad()
def neg_log_likelihood(model, x, L=64):
    """IWAE bound with L posterior samples. Lower = more normal."""
    mu, logvar = model.encode(x)                        # (B,k)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(L, *mu.shape, device=x.device)    # (L,B,k)
    z = mu + std * eps
    B, k = mu.shape

    x_hat = model.decode(z.reshape(L * B, k)).reshape(L, B, -1)
    lv = model.log_var_x

    log_p_x_z = -0.5 * (((x - x_hat) ** 2) / lv.exp() + lv + LOG2PI).sum(-1)
    log_p_z = -0.5 * (z ** 2 + LOG2PI).sum(-1)
    log_q_z = -0.5 * (eps ** 2 + logvar + LOG2PI).sum(-1)

    log_w = log_p_x_z + log_p_z - log_q_z               # (L,B)
    return -(torch.logsumexp(log_w, 0) - math.log(L))   # (B,)


@torch.no_grad()
def score_all(model, X, batch=1024, L=64):
    model.eval()
    return torch.cat([neg_log_likelihood(model, X[i:i + batch].to(DEVICE), L).cpu()
                      for i in range(0, len(X), batch)])


@torch.no_grad()
def per_dim_kl(model, X, n=2048):
    """Diagnostic: a dimension whose KL is ~0 has collapsed to the prior."""
    mu, lv = model.encode(X[:n].to(DEVICE))
    return (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).mean(0).cpu().tolist()


def roc_auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ============================================================================
# training
# ============================================================================

class Splits(TypedDict):
    va: torch.Tensor
    te_n: torch.Tensor
    te_a: torch.Tensor
    cols: list[str]
    best_ep: int


def train_vae(data, epochs=40, warmup=8, hidden=(128, 64), latent=12, lr=1e-3,
              batch=512, patience_max=8, diag_n=4000, diag=True, quiet=False):
    """Train on legitimate rows only.

    Three changes from the notebook loop, all of which leave the maths alone:

    1. the training split lives on the GPU for the whole run, so the 389
       batches/epoch are sliced on-device instead of being copied across PCIe
    2. nothing calls .item() inside the step loop -- the notebook synced three
       times per step (~47k stalls over 40 epochs), each costing more than the
       microseconds of compute it waited for
    3. shuffling is a device-side randperm rather than a DataLoader

    On a 26k-parameter model these dominate the runtime; without them a GPU sits
    at ~0% utilisation and loses to the CPU.
    """
    X_legit, X_fraud, cols = data
    D = X_legit.shape[1]

    n_tr, n_va = int(.7 * len(X_legit)), int(.15 * len(X_legit))
    tr_r, va_r = X_legit[:n_tr], X_legit[n_tr:n_tr + n_va]
    te_r = X_legit[n_tr + n_va:]

    mu_, sd_ = tr_r.mean(0), tr_r.std(0) + 1e-8        # TRAIN stats only
    z = lambda a: torch.from_numpy((a - mu_) / sd_)
    tr, va, te_n, te_a = z(tr_r), z(va_r), z(te_r), z(X_fraud)

    if not quiet:
        print(f"train {len(tr)} | val {len(va)} | test-legit {len(te_n)} "
              f"| test-fraud {len(te_a)}")
        steps = -(-len(tr) // batch)          # ceil, so it stays right if n % batch == 0
        print(f"{steps} steps/epoch -> {steps * epochs} total updates")

    tr = tr.to(DEVICE)                                 # 23 MB, stays resident
    model = VAE(D, hidden=hidden, latent=latent).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    keys = ("epoch", "beta", "train", "nll", "kl", "val", "sigma_x", "auc", "ap")
    hist: dict[str, list] = {k: [] for k in keys}
    best, best_state, best_ep, patience = float("inf"), None, 0, 0

    dn = te_n[:diag_n]
    dlab = np.r_[np.zeros(len(dn)), np.ones(len(te_a))]
    n = len(tr)

    for ep in range(1, epochs + 1):
        model.train()
        beta = min(1.0, ep / warmup)

        # accumulate on device; a single sync happens after the epoch
        tot = torch.zeros((), device=DEVICE)
        tn = torch.zeros((), device=DEVICE)
        tk = torch.zeros((), device=DEVICE)
        nb = 0

        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            xb = tr[perm[i:i + batch]]
            x_hat, m, lv = model(xb)
            loss, nll, kl = elbo_loss(xb, x_hat, m, lv, model.log_var_x, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.detach(); tn += nll; tk += kl; nb += 1

        tot_f, tn_f, tk_f = (tot / nb).item(), (tn / nb).item(), (tk / nb).item()
        val = score_all(model, va, L=8).mean().item()
        sig = float(model.log_var_x.detach().exp().sqrt())

        # AUC*/AP* use fraud labels: DIAGNOSTIC ONLY, never for model selection.
        # AP is the one to watch -- see finding 3 in README.md.
        if diag:
            ds = np.r_[score_all(model, dn, L=8).numpy(),
                       score_all(model, te_a, L=8).numpy()]
            auc = roc_auc(ds, dlab)
            ap = average_precision_score(dlab, ds)
        else:
            auc = ap = float("nan")

        for k, v in zip(keys, (ep, beta, tot_f, tn_f, tk_f, val, sig, auc, ap)):
            hist[k].append(v)
        if not quiet:
            print(f"ep {ep:3d} | beta {beta:.2f} | train {tot_f:9.2f} | nll {tn_f:8.2f} "
                  f"| kl {tk_f:5.2f} | val -logp {val:8.2f} | sigma_x {sig:.3f} "
                  f"| AUC* {auc:.4f} | AP* {ap:.4f}")

        if ep > warmup:
            if val < best - 1e-3:
                best, best_ep, patience = val, ep, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= patience_max:
                    if not quiet:
                        print(f"early stop at epoch {ep}")
                    break

    if best_state:
        model.load_state_dict(best_state)
        if not quiet:
            print(f"restored best epoch {best_ep} (val -logp {best:.2f})")

    return model, hist, Splits(va=va, te_n=te_n, te_a=te_a, cols=cols, best_ep=best_ep)


def evaluate(model, sp: Splits, L=64):
    s = {"va":   score_all(model, sp["va"],   L=L).numpy(),
         "te_n": score_all(model, sp["te_n"], L=L).numpy(),
         "te_a": score_all(model, sp["te_a"], L=L).numpy()}
    s["scores"] = np.r_[s["te_n"], s["te_a"]]
    s["labels"] = np.r_[np.zeros(len(s["te_n"])), np.ones(len(s["te_a"]))]
    return s


# ============================================================================
# figures
# ============================================================================

def _save(fig, figdir: Path, name: str):
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path}")


def plot_overview(Xl, Xf, cols, figdir: Path):
    rng = np.random.default_rng(SEED)
    sub = Xl[rng.choice(len(Xl), 8000, replace=False)]
    mu_, sd_ = sub.mean(0), sub.std(0) + 1e-8
    p = PCA(n_components=2, random_state=SEED).fit((sub - mu_) / sd_)
    Zl, Zf = p.transform((sub - mu_) / sd_), p.transform((Xf - mu_) / sd_)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(*Zl.T, s=5, c=C_LEGIT, alpha=.25, lw=0, label="Legitimate")
    ax[0].scatter(*Zf.T, s=30, c=C_FRAUD, marker="^", lw=.8,
                  edgecolor=SURFACE, label="Fraud")
    ax[0].set(title="Transactions in the first two principal components",
              xlabel=f"PC1 ({p.explained_variance_ratio_[0]:.0%} var)",
              ylabel=f"PC2 ({p.explained_variance_ratio_[1]:.0%} var)")
    ax[0].grid(axis="both"); ax[0].legend(loc="upper right")

    bins = np.linspace(0, np.log1p(3000), 60)
    ax[1].hist(Xl[:, cols.index("Amount")], bins=bins, density=True,
               color=C_LEGIT, alpha=.75, label="Legitimate")
    ax[1].hist(Xf[:, cols.index("Amount")], bins=bins, density=True,
               color=C_FRAUD, alpha=.75, label="Fraud")
    ax[1].set(title="Transaction amount, log1p scale",
              xlabel="log1p(Amount)", ylabel="density")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, figdir, "data_overview.png")


def plot_curves(hist, figdir: Path):
    e = hist["epoch"]
    panels = [("train", "ELBO loss (train)", "loss"),
              ("nll", "Reconstruction NLL", "nats"),
              ("kl", "KL to prior", "nats"),
              ("val", "Val -log p(x), legit only", "nats"),
              ("auc", "Detection vs fraud (diagnostic)", "metric"),
              ("sigma_x", "Learned observation noise sigma_x", "sigma")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7))
    for ax, (k, title, ylab) in zip(axes.ravel(), panels):
        if k == "auc":
            ax.plot(e, hist["auc"], color=C_LEGIT, label="AUC*")
            ax.plot(e, hist["ap"], color=C_FRAUD, label="AP*")
            ax.set(title=title, xlabel="epoch", ylabel=ylab)
            if not np.all(np.isnan(hist["ap"])):
                pk = e[int(np.nanargmax(hist["ap"]))]
                ax.axvline(pk, color=C_FRAUD, ls="--", lw=1.2)
                ax.annotate(f"peak AP (ep {pk})", (pk, np.nanmin(hist["ap"])),
                            color=C_FRAUD, fontsize=9, ha="left", va="bottom",
                            xytext=(4, 2), textcoords="offset points")
            ax.legend(loc="lower right")
            continue

        ax.plot(e, hist[k], color=C_LEGIT)
        ax.set(title=title, xlabel="epoch", ylabel=ylab)
        if k == "val":
            b = e[int(np.argmin(hist["val"]))]
            ax.axvline(b, color=C_THR, ls="--", lw=1.2)
            ax.annotate(f"best val (ep {b})", (b, max(hist["val"])),
                        color=C_THR, fontsize=9, ha="left", va="top",
                        xytext=(4, -2), textcoords="offset points")
    fig.suptitle("VAE training diagnostics — legit transactions only",
                 x=.008, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    _save(fig, figdir, "training_curves.png")


def plot_detection(model, sp, S, figdir: Path, fpr_budget=0.001):
    thr = np.percentile(S["va"], 100 * (1 - fpr_budget))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    lo, hi = np.percentile(S["scores"], [.5, 99.9])
    bins = np.linspace(lo, hi, 80)
    ax[0, 0].hist(S["te_n"], bins=bins, color=C_LEGIT, alpha=.8, label="Legitimate")
    ax[0, 0].hist(S["te_a"], bins=bins, color=C_FRAUD, alpha=.8, label="Fraud")
    ax[0, 0].axvline(thr, color=C_THR, ls="--", lw=1.4)
    ax[0, 0].set(yscale="log", title="Anomaly score distribution",
                 xlabel="-log p(x)", ylabel="count (log)")
    # pinned to the axes top so the label never lands on the bars
    ax[0, 0].annotate(f"threshold {thr:.1f}", xy=(thr, 1),
                      xycoords=("data", "axes fraction"),
                      xytext=(5, -4), textcoords="offset points",
                      color=C_THR, fontsize=9, ha="left", va="top")
    ax[0, 0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.92))

    fpr, tpr, _ = roc_curve(S["labels"], S["scores"])
    ax[0, 1].plot(fpr, tpr, color=C_LEGIT,
                  label=f"VAE (AUC {roc_auc_score(S['labels'], S['scores']):.4f})")
    ax[0, 1].plot([0, 1], [0, 1], color=AXLINE, lw=1, label="Chance")
    ax[0, 1].set(title="ROC — optimistic @ 0.17% prevalence",
                 xlabel="false positive rate", ylabel="true positive rate")
    ax[0, 1].grid(axis="both"); ax[0, 1].legend(loc="lower right")

    pr, rc, _ = precision_recall_curve(S["labels"], S["scores"])
    ap = average_precision_score(S["labels"], S["scores"])
    base = S["labels"].mean()
    ax[1, 0].plot(rc, pr, color=C_FRAUD, label=f"VAE (AP {ap:.4f})")
    ax[1, 0].axhline(base, color=AXLINE, lw=1, label=f"Chance ({base:.4f})")
    ax[1, 0].set(title="Precision-recall", xlabel="recall", ylabel="precision")
    ax[1, 0].grid(axis="both"); ax[1, 0].legend(loc="upper right")

    with torch.no_grad():
        mn = model.encode(sp["te_n"][:6000].to(DEVICE))[0].cpu().numpy()
        ma = model.encode(sp["te_a"].to(DEVICE))[0].cpu().numpy()
    q = PCA(n_components=2, random_state=SEED).fit(mn)
    An, Aa = q.transform(mn), q.transform(ma)
    ax[1, 1].scatter(*An.T, s=5, c=C_LEGIT, alpha=.25, lw=0, label="Legitimate")
    ax[1, 1].scatter(*Aa.T, s=30, c=C_FRAUD, marker="^", lw=.8,
                     edgecolor=SURFACE, label="Fraud")
    ax[1, 1].set(title="Latent posterior means, 2D projection",
                 xlabel="latent PC1", ylabel="latent PC2")
    ax[1, 1].grid(axis="both"); ax[1, 1].legend()

    fig.tight_layout()
    _save(fig, figdir, "detection.png")
    return thr


def plot_feature_error(model, sp, S, figdir: Path, K=25):
    # .copy() because torch cannot index with a negative-stride array
    idx = np.argsort(S["te_a"])[-K:][::-1].copy()
    with torch.no_grad():
        xa = sp["te_a"][idx].to(DEVICE)
        err = ((xa - model.decode(model.encode(xa)[0])) ** 2).cpu().numpy()

    fig, ax = plt.subplots(figsize=(13, 5.5))
    im = ax.imshow(err, aspect="auto", cmap=BLUE_RAMP,
                   vmin=0, vmax=np.percentile(err, 99))
    ax.set(title=f"Per-feature squared reconstruction error — {K} highest-scoring frauds",
           xlabel="feature", ylabel="rank by anomaly score")
    ax.set_xticks(range(len(sp["cols"])), sp["cols"], rotation=90, fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=.015)
    plt.setp(cb.outline, visible=False)
    cb.set_label("squared error", color=INK2)
    fig.tight_layout()
    _save(fig, figdir, "feature_error.png")


# ============================================================================
# reporting
# ============================================================================

def threshold_table(S):
    print(f"\n{'pct':>6} {'thr':>9} {'recall':>7} {'prec':>7} {'F1':>6} {'TP':>5} {'FP':>6}")
    for p in (95, 99, 99.5, 99.9, 99.95, 99.99):
        t = np.percentile(S["va"], p)
        tp, fp = int((S["te_a"] > t).sum()), int((S["te_n"] > t).sum())
        rec = tp / len(S["te_a"])
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{p:6.2f} {t:9.2f} {rec:7.3f} {prec:7.3f} {f1:6.3f} {tp:5d} {fp:6d}")


def isolation_forest_baseline(sp, S):
    """One-class baseline, fit on the same held-out legit rows the VAE
    calibrates its threshold on.

    max_samples: sklearn's 'auto' default is min(256, n) -- only 256 of 42,647
    rows per tree, which badly underfits 29 dimensions. 16384 is the fair
    setting, and the difference is almost entirely in AP, not AUC.
    """
    iso = IsolationForest(n_estimators=300, contamination="auto",
                          random_state=SEED, n_jobs=-1,
                          max_samples=16384).fit(sp["va"].numpy())
    b = -np.r_[iso.score_samples(sp["te_n"].numpy()),
               iso.score_samples(sp["te_a"].numpy())]
    print(f"\nVAE     AUC {roc_auc_score(S['labels'], S['scores']):.4f} | "
          f"AP {average_precision_score(S['labels'], S['scores']):.4f}")
    print(f"iForest AUC {roc_auc_score(S['labels'], b):.4f} | "
          f"AP {average_precision_score(S['labels'], b):.4f}")


# ============================================================================
# tuning
# ============================================================================
# Hyperparameters cannot be chosen on validation -log p(x): density fit and
# detection are nearly decoupled, and more capacity always fits the legit
# density better while detection stays flat. So the labelled test data is split
# 40% SELECTION / 60% FINAL with a fixed mask; configs are ranked on the
# selection slice and the winner is reported on the untouched final slice.

SEL_FRAC = 0.40
MASK_SEED = 20240930          # deliberately distinct from SEED


def selection_masks(data, quiet=False):
    n_legit = len(data[0])
    n_te_n = n_legit - int(.7 * n_legit) - int(.15 * n_legit)
    n_te_a = len(data[1])
    rs = np.random.default_rng(MASK_SEED)
    mask_n = rs.random(n_te_n) < SEL_FRAC      # True -> selection
    mask_a = rs.random(n_te_a) < SEL_FRAC
    if not quiet:
        print(f"selection slice: {mask_n.sum():>6} legit + {mask_a.sum():>3} fraud")
        print(f"final slice    : {(~mask_n).sum():>6} legit + {(~mask_a).sum():>3} fraud")
    return mask_n, mask_a


def _ap_auc(scores_n, scores_a):
    s = np.r_[scores_n, scores_a]
    y = np.r_[np.zeros(len(scores_n)), np.ones(len(scores_a))]
    return average_precision_score(y, s), roc_auc_score(y, s)


def run_config(data, masks, latent=12, hidden=(128, 64), lr=1e-3, batch=512,
               warmup=8, epochs=25, seed=None, L=64):
    """One training run, silenced. Returns a dict of metrics.

    seed defaults to None rather than SEED because a default argument binds at
    definition time -- it would pin every sweep run to 42 and silently ignore
    --seed.
    """
    mask_n, mask_a = masks
    seed = SEED if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model, hist, sp = train_vae(data, epochs=epochs, warmup=warmup,
                                    hidden=hidden, latent=latent, lr=lr,
                                    batch=batch, diag=False, quiet=True)
        S = evaluate(model, sp, L=L)
        kl_dims = per_dim_kl(model, sp["va"])

    ap_sel, _ = _ap_auc(S["te_n"][mask_n], S["te_a"][mask_a])
    ap_fin, auc_fin = _ap_auc(S["te_n"][~mask_n], S["te_a"][~mask_a])
    return dict(latent=latent, hidden=hidden, lr=lr, batch=batch, epochs=epochs,
                params=sum(p.numel() for p in model.parameters()),
                val=min(hist["val"]), best_ep=sp["best_ep"],
                alive=sum(v > 0.05 for v in kl_dims),
                ap_sel=ap_sel, ap_final=ap_fin, auc_final=auc_fin,
                secs=time.perf_counter() - t0)


def cmd_sweep(args, data):
    masks = selection_masks(data)
    grid: list[dict[str, Any]] = ([dict(latent=k) for k in (4, 8, 12, 16, 24, 32)] +
                                  [dict(latent=16, hidden=(256, 128)),
                                   dict(latent=16, hidden=(64, 32))])
    res = []
    for i, cfg in enumerate(grid, 1):
        r = run_config(data, masks, epochs=args.epochs, batch=args.batch, **cfg)
        res.append(r)
        print(f"[{i}/{len(grid)}] latent={r['latent']:<3} "
              f"hidden={str(r['hidden']):<11} alive={r['alive']:>2}/{r['latent']:<3} "
              f"val={r['val']:8.2f} AP_sel={r['ap_sel']:.4f}  ({r['secs']:.0f}s)")

    df = pd.DataFrame(res).sort_values("ap_sel", ascending=False)
    print("\n--- ranked by SELECTION-slice AP (the honest criterion) ---")
    print(df[["latent", "hidden", "params", "alive", "val", "best_ep",
              "ap_sel", "ap_final", "auc_final"]].to_string(index=False))

    print("\nval -logp ranks configs by density fit, which rewards capacity:")
    print(df.sort_values("val")[["latent", "hidden", "val", "ap_final"]]
          .to_string(index=False))

    best_ap = df.iloc[0]
    best_val = df.sort_values("val").iloc[0]
    base = next(r for r in res if r["latent"] == 12 and r["hidden"] == (128, 64))
    print(f"\n{'selector':<26} {'config':<26} {'FINAL AP':>9} {'FINAL AUC':>10}")
    for name, r in (("SELECTION-slice AP", best_ap),
                    ("val -logp only", best_val),
                    ("notebook default", base)):
        cfg = f"latent={r['latent']} hidden={r['hidden']}"
        print(f"{name:<26} {cfg:<26} {r['ap_final']:>9.4f} {r['auc_final']:>10.4f}")
    return df


def cmd_fine_sweep(args, data):
    """latent 18-22 with 12/24 anchors, several seeds.

    The coarse sweep's single-seed gaps (~0.02 AP) sit at the noise floor, so
    one run cannot rank these. The anchors share the seeds, so the comparison
    is like-for-like.
    """
    masks = selection_masks(data)
    fine = []
    for lat in args.latents:
        for sd in args.seeds:
            r = run_config(data, masks, epochs=args.epochs, latent=lat,
                           batch=args.batch, seed=sd)
            r["seed"] = sd
            fine.append(r)
            print(f"latent {r['latent']:>2} seed {sd} | alive {r['alive']:>2}/{r['latent']:<2} "
                  f"| val {r['val']:7.2f} | AP_sel {r['ap_sel']:.4f} "
                  f"| AP_fin {r['ap_final']:.4f} | AUC {r['auc_final']:.4f}  ({r['secs']:.0f}s)")

    fdf = pd.DataFrame(fine)
    agg = fdf.groupby("latent").agg(
        alive=("alive", "mean"), val=("val", "mean"),
        ap_sel=("ap_sel", "mean"), ap_sel_sd=("ap_sel", "std"),
        ap_final=("ap_final", "mean"), ap_final_sd=("ap_final", "std"),
        auc=("auc_final", "mean")).round(4)
    agg.insert(1, "dead", agg.index - agg["alive"])
    print(f"\n--- mean over {len(args.seeds)} seeds ---")
    print(agg.to_string())

    pooled_sd = fdf.groupby("latent").ap_final.std().mean()
    print(f"\npooled across-seed AP sd: {pooled_sd:.4f}  "
          f"-> gaps smaller than ~{2*pooled_sd:.3f} are not evidence")

    best_l = agg.ap_sel.idxmax()
    print(f"\nbest by SELECTION AP : latent={best_l}  -> "
          f"FINAL AP {agg.loc[best_l, 'ap_final']:.4f} "
          f"+/- {agg.loc[best_l, 'ap_final_sd']:.4f}")
    for anchor in (12, 24):
        if anchor in agg.index:
            print(f"latent={anchor} anchor     :  FINAL AP {agg.loc[anchor, 'ap_final']:.4f}")
    return agg


def cmd_train(args, data, baseline=False):
    Xl, Xf, cols = data
    print(f"legit {Xl.shape}  fraud {Xf.shape}  "
          f"prevalence {len(Xf)/(len(Xl)+len(Xf)):.4%}")

    if args.figures:
        plot_overview(Xl, Xf, cols, args.figdir)

    model, hist, sp = train_vae(data, epochs=args.epochs, warmup=args.warmup,
                                hidden=tuple(args.hidden), latent=args.latent,
                                lr=args.lr, batch=args.batch,
                                patience_max=args.patience, diag=args.diag)

    kl_dims = per_dim_kl(model, sp["va"])
    print("\nKL per latent dim:", [round(v, 2) for v in kl_dims])
    alive = sum(v > 0.05 for v in kl_dims)
    print(f"alive dims: {alive}/{args.latent}  "
          f"({'search higher' if alive == args.latent else 'at or past the bottleneck'})")

    S = evaluate(model, sp)
    print(f"\nAUC {roc_auc_score(S['labels'], S['scores']):.4f} | "
          f"AP {average_precision_score(S['labels'], S['scores']):.4f}")
    threshold_table(S)

    if args.figures:
        plot_curves(hist, args.figdir)
        plot_detection(model, sp, S, args.figdir)
        plot_feature_error(model, sp, S, args.figdir)

    if baseline:
        isolation_forest_baseline(sp, S)

    if args.save:
        torch.save({"state_dict": model.state_dict(),
                    "latent": args.latent, "hidden": tuple(args.hidden),
                    "cols": sp["cols"]}, args.save)
        print(f"\nmodel -> {args.save}")

    return model, hist, sp, S


# ============================================================================
# cli
# ============================================================================

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="where creditcard.csv lives (default: data)")
    common.add_argument("--device", default=None,
                        help="cuda / cpu (default: cuda when available)")
    common.add_argument("--seed", type=int, default=SEED)
    common.add_argument("--epochs", type=int, default=40)
    common.add_argument("--batch", type=int, default=512,
                        help="512 matches the notebook; 8192+ is much faster on a GPU")

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_train_args(sp_):
        sp_.add_argument("--latent", type=int, default=12)
        sp_.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
        sp_.add_argument("--lr", type=float, default=1e-3)
        sp_.add_argument("--warmup", type=int, default=8)
        sp_.add_argument("--patience", type=int, default=8)
        sp_.add_argument("--figdir", type=Path, default=Path("figures"))
        sp_.add_argument("--no-figures", dest="figures", action="store_false")
        sp_.add_argument("--no-diag", dest="diag", action="store_false",
                         help="skip the per-epoch AUC*/AP* diagnostic (it peeks at labels)")
        sp_.add_argument("--save", type=Path, default=None,
                         help="write the trained weights here")

    t = sub.add_parser("train", parents=[common], help="train one configuration")
    add_train_args(t)

    b = sub.add_parser("baseline", parents=[common],
                       help="train, then compare against Isolation Forest")
    add_train_args(b)

    s = sub.add_parser("sweep", parents=[common], help="coarse hyperparameter grid")
    s.set_defaults(epochs=25)

    f = sub.add_parser("fine-sweep", parents=[common],
                       help="latent 18-22 with 12/24 anchors, multi-seed")
    f.add_argument("--latents", type=int, nargs="+",
                   default=[12, 18, 19, 20, 21, 22, 24])
    f.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    f.set_defaults(epochs=25)

    return p


def main(argv=None):
    global DEVICE, SEED

    args = build_parser().parse_args(argv)

    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {DEVICE}", end="")
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        print(f"  ({torch.cuda.get_device_name(0)})")
    elif DEVICE.startswith("cuda"):
        print("  -- requested but torch.cuda.is_available() is False", file=sys.stderr)
        sys.exit(1)
    else:
        print()

    csv_path = resolve_data(args.data_dir)
    print(f"dataset: {csv_path}  ({csv_path.stat().st_size / 1e6:.1f} MB)")
    data = make_data(csv_path)

    if args.cmd == "train":
        cmd_train(args, data)
    elif args.cmd == "baseline":
        cmd_train(args, data, baseline=True)
    elif args.cmd == "sweep":
        cmd_sweep(args, data)
    elif args.cmd == "fine-sweep":
        cmd_fine_sweep(args, data)


if __name__ == "__main__":
    main()
