"""All figures in one place. Every function reads results/ and writes a PNG."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

NEG, POS = "#D55E00", "#0072B2"
CLUSTER_COLORS = ["#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999", "#F0E442", "#8E44AD", "#2C3E50"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 170,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25})


class Results:
    def __init__(self, root: Path):
        self.root, self.res = Path(root), Path(root) / "results"
        self.fig = self.res / "figures"
        self.fig.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(self.res / "summary.csv")
        self.terms = json.loads((self.res / "cluster_terms.json").read_text())
        self.var = json.loads((self.res / "svd_variance.json").read_text())

    def sel(self, method, **kw):
        d = self.df[self.df.method == method]
        for k, v in kw.items():
            d = d[d[k] == v]
        return d

    def emb(self, method, n, p):
        tag = "sigma1" if p == "sigma1" else f"p{int(p)}"
        return np.load(self.res / "embeddings" / f"{method.replace('_sigma1', '')}_n{n}_{tag}.npy")

    def labels(self, n):
        return pd.read_csv(self.res / "clusters" / f"kmedoids_n{n}_k3.csv")["sentiment"].to_numpy()


def _scatter(ax, Y, lab, n):
    s = 14 if n <= 500 else 7 if n <= 2500 else 2 if n <= 10000 else 1
    a = 0.8 if n <= 2500 else 0.5
    for v, c in ((0, NEG), (1, POS)):
        m = lab == v
        ax.scatter(Y[m, 0], Y[m, 1], s=s, c=c, alpha=a, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def _sent_legend(fig, y=0.0):
    h = [Line2D([], [], marker="o", ls="", color=NEG, label="Negative review"),
         Line2D([], [], marker="o", ls="", color=POS, label="Positive review")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, y))


def perplexity_grid(R, n, method="tsne"):
    perps = sorted(R.sel(method, n=n).perplexity.dropna().unique())
    cols = 3
    rows = int(np.ceil(len(perps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.5 * rows))
    lab = R.labels(n)
    for ax, p in zip(np.ravel(axes), perps):
        r = R.sel(method, n=n, perplexity=p).iloc[0]
        _scatter(ax, R.emb(method, n, p), lab, n)
        ax.set_title(f"perplexity {int(p)}\ntrust {r.trustworthiness:.3f} · kNN keep {r.knn_preservation:.3f} · "
                     f"label agr. {r.label_agreement_2d:.3f}", fontsize=8)
    name = "t-SNE" if method == "tsne" else "SNE"
    fig.suptitle(f"{name}, n = {n:,} reviews: effect of perplexity", y=1.01, fontsize=11)
    _sent_legend(fig, -0.03)
    out = R.fig / f"{method}_perplexity_grid_n{n}.png"
    fig.savefig(out); plt.close(fig); return out


def size_grid(R, p=30):
    sizes = sorted(R.sel("tsne", perplexity=p).n.unique())
    cols = 4
    rows = int(np.ceil(len(sizes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    for ax, n in zip(np.ravel(axes), sizes):
        r = R.sel("tsne", n=n, perplexity=p).iloc[0]
        _scatter(ax, R.emb("tsne", n, p), R.labels(n), n)
        ax.set_title(f"n = {n:,}\ntrust {r.trustworthiness:.3f} · label agr. {r.label_agreement_2d:.3f}", fontsize=8)
    for ax in np.ravel(axes)[len(sizes):]:
        ax.axis("off")
    fig.suptitle(f"t-SNE at perplexity {p}: effect of sample size", y=1.01, fontsize=11)
    _sent_legend(fig, -0.03)
    out = R.fig / f"tsne_size_grid_p{p}.png"
    fig.savefig(out); plt.close(fig); return out


def heatmaps(R):
    d = R.sel("tsne").copy()
    d["label_gap"] = d.label_agreement_2d - d.label_agreement_hd
    spec = [("trustworthiness", "Trustworthiness (k=10)", "viridis", "{:.3f}"),
            ("knn_preservation", "kNN preservation (k=10)", "viridis", "{:.3f}"),
            ("label_agreement_2d", "Same-sentiment share of 10 nearest 2-D neighbours", "viridis", "{:.3f}"),
            ("kl", "Final KL divergence", "magma_r", "{:.2f}"),
            ("seconds", "Run time (s)", "magma_r", "{:.0f}"),
            ("label_gap", "Label agreement: 2-D map minus 50-D space", "RdBu", "{:+.3f}")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, (col, title, cmap, fmt) in zip(axes.ravel(), spec):
        pv = d.pivot(index="n", columns="perplexity", values=col).sort_index(ascending=False)
        im = ax.imshow(pv.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(pv.shape[1]), [int(c) for c in pv.columns])
        ax.set_yticks(range(pv.shape[0]), [f"{int(i):,}" for i in pv.index])
        ax.set_xlabel("Perplexity"); ax.set_ylabel("Sample size n"); ax.set_title(title); ax.grid(False)
        lo, hi = np.nanmin(pv.values), np.nanmax(pv.values)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isfinite(v):
                    dark = (v - lo) / (hi - lo + 1e-12) < 0.5
                    dark = dark if cmap in ("viridis", "RdBu") else not dark
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                            color="white" if dark else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("t-SNE: quality and cost across the full perplexity × sample-size grid", fontsize=12)
    fig.tight_layout()
    out = R.fig / "tsne_heatmaps.png"
    fig.savefig(out); plt.close(fig); return out


def metric_lines(R):
    d = R.sel("tsne")
    sizes = sorted(d.n.unique())
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(sizes)))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, t) in zip(axes[0], [("trustworthiness", "Trustworthiness"), ("knn_preservation", "kNN preservation"),
                                      ("label_agreement_2d", "Label agreement (2-D)")]):
        for c, n in zip(cmap, sizes):
            s = d[d.n == n].sort_values("perplexity")
            ax.plot(s.perplexity, s[col], "o-", color=c, ms=4, label=f"n={n:,}")
        ax.set_xlabel("Perplexity"); ax.set_title(f"{t} vs perplexity")
    axes[0, 0].legend(fontsize=7, ncol=2)
    for ax, (col, t) in zip(axes[1], [("trustworthiness", "Trustworthiness"), ("knn_preservation", "kNN preservation"),
                                      ("label_agreement_2d", "Label agreement (2-D)")]):
        for p, c in zip(sorted(d.perplexity.unique()), plt.cm.plasma(np.linspace(0, 0.85, 6))):
            s = d[d.perplexity == p].sort_values("n")
            ax.plot(s.n, s[col], "o-", color=c, ms=4, label=f"perp {int(p)}")
        if col == "label_agreement_2d":
            hd = d.drop_duplicates("n").sort_values("n")
            ax.plot(hd.n, hd.label_agreement_hd, "k--", label="original 50-D space")
        ax.set_xscale("log"); ax.set_xlabel("Sample size n"); ax.set_title(f"{t} vs sample size")
    axes[1, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = R.fig / "tsne_metric_lines.png"
    fig.savefig(out); plt.close(fig); return out


def runtime(R):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    series = [("tsne", 30, "t-SNE (Barnes-Hut), perp 30", "o-", "#0072B2"),
              ("tsne", 5, "t-SNE (Barnes-Hut), perp 5", "o--", "#56B4E9"),
              ("sne", 30, "Exact SNE, perp 30", "s-", "#D55E00"),
              ("kmedoids", None, "K-medoids (PAM), k=3", "^-", "#009E73")]
    for m, p, lab, st, c in series:
        s = R.sel(m, k=3) if m == "kmedoids" else R.sel(m, perplexity=p)
        s = s.sort_values("n")
        ax.loglog(s.n, s.seconds, st, color=c, label=lab, ms=5)
    n = np.array([250, 25000.0])
    t0 = R.sel("tsne", perplexity=30, n=250).seconds.iloc[0]
    ax.loglog(n, t0 * (n / 250) * np.log(n) / np.log(250), ":", color="grey", label="n log n guide")
    s0 = R.sel("sne", perplexity=30, n=250).seconds.iloc[0]
    ax.loglog(n, s0 * (n / 250) ** 2, "-.", color="grey", lw=0.8, label="n² guide")
    ax.set_xlabel("Sample size n"); ax.set_ylabel("Seconds (1 CPU core)")
    ax.set_title("Run time scaling"); ax.legend(fontsize=7)
    out = R.fig / "runtime_scaling.png"
    fig.savefig(out); plt.close(fig); return out


def sne_vs_tsne(R, n=1000, p=30):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    lab = R.labels(n)
    items = [("sne_sigma1", "sigma1", "Original SNE (σ = 1)"), ("sne", p, f"SNE, perplexity {p}"),
             ("tsne", p, f"t-SNE, perplexity {p}")]
    for ax, (m, pp, t) in zip(axes, items):
        r = R.sel(m, n=n) if m == "sne_sigma1" else R.sel(m, n=n, perplexity=pp)
        r = r.iloc[0]
        _scatter(ax, R.emb(m, n, pp), lab, n)
        extra = f"\neffective perplexity {r.eff_perplexity:.0f} of {n-1}" if m == "sne_sigma1" else ""
        ax.set_title(f"{t}\ntrust {r.trustworthiness:.3f} · label agr. {r.label_agreement_2d:.3f}{extra}", fontsize=8)
    fig.suptitle(f"Original SNE vs calibrated SNE vs t-SNE (n = {n:,})", y=1.03, fontsize=11)
    _sent_legend(fig, -0.06)
    out = R.fig / f"sne_vs_tsne_n{n}.png"
    fig.savefig(out); plt.close(fig); return out


def sne_quality(R):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    s1 = R.sel("sne_sigma1").sort_values("n")
    axes[0].bar(range(len(s1)), s1.n - 1, color="#cccccc", label="maximum possible (n−1)")
    axes[0].bar(range(len(s1)), s1.eff_perplexity, color="#D55E00", width=0.5, label="σ = 1 effective perplexity")
    axes[0].set_xticks(range(len(s1)), [f"{int(n):,}" for n in s1.n]); axes[0].set_xlabel("n")
    axes[0].set_title("σ = 1 makes P almost uniform"); axes[0].legend(fontsize=7)
    sizes = sorted(R.sel("sne").n.unique())
    for ax, (col, t) in zip(axes[1:], [("trustworthiness", "Trustworthiness"), ("label_agreement_2d", "Label agreement (2-D)")]):
        for c, n in zip(plt.cm.viridis(np.linspace(0, 0.9, len(sizes))), sizes):
            s = R.sel("sne", n=n).sort_values("perplexity")
            ax.plot(s.perplexity, s[col], "o-", color=c, ms=4, label=f"SNE n={n:,}")
            b = R.sel("sne_sigma1", n=n)[col].iloc[0]
            ax.plot([5], [b], "x", color=c, ms=7)
        ax.set_xlabel("Perplexity"); ax.set_title(f"Exact SNE: {t}  (✕ = σ = 1)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    out = R.fig / "sne_quality.png"
    fig.savefig(out); plt.close(fig); return out


def kmedoids_overview(R):
    km = R.sel("kmedoids", k=3).sort_values("n")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    axes[0].plot(km.n, km.cost_per_point, "o-", color="#009E73"); axes[0].set_title("Mean distance to medoid")
    axes[1].plot(km.n, km.silhouette, "o-", color="#CC79A7"); axes[1].set_title("Silhouette (SVD space)")
    axes[2].plot(km.n, km.ari_vs_sentiment, "o-", color="#E69F00", label="ARI")
    axes[2].plot(km.n, km.nmi_vs_sentiment, "s-", color="#56B4E9", label="NMI")
    axes[2].set_title("Agreement with sentiment"); axes[2].legend(fontsize=7)
    for ax in axes[:3]:
        ax.set_xscale("log"); ax.set_xlabel("Sample size n")
    ks = R.sel("kmedoids", n=2500).sort_values("k")
    ax = axes[3]; ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax.plot(ks.k, ks.silhouette, "o-", color="#CC79A7"); ax.set_ylabel("Silhouette", color="#CC79A7")
    ax2.plot(ks.k, ks.cost_per_point, "s--", color="#009E73"); ax2.set_ylabel("Cost / point", color="#009E73")
    ax.set_xlabel("k"); ax.set_title("Choosing k (n = 2,500)")
    fig.tight_layout()
    out = R.fig / "kmedoids_overview.png"
    fig.savefig(out); plt.close(fig); return out


def cluster_profile(R, n):
    cl = pd.read_csv(R.res / "clusters" / f"kmedoids_n{n}_k3.csv")
    Y = R.emb("tsne", n, 30)
    info = R.terms[str(n)]
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.8, 1.3])
    ax = fig.add_subplot(gs[0])
    s = 6 if n <= 5000 else 1.5
    for c in range(3):
        m = cl.cluster.to_numpy() == c
        ax.scatter(Y[m, 0], Y[m, 1], s=s, c=CLUSTER_COLORS[c], alpha=0.6, linewidths=0,
                   label=f"Cluster {c} (n={m.sum():,})", rasterized=True)
        dd = np.where(m, cl.dist_to_medoid.to_numpy(), np.inf)
        med = [int(np.argmin(dd))]
        ax.scatter(Y[med, 0], Y[med, 1], marker="*", s=260, c=CLUSTER_COLORS[c], edgecolors="black", linewidths=1)
    ax.legend(markerscale=4, fontsize=8, loc="best"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"K-medoids clusters on the t-SNE map (perp 30, n={n:,}); ★ = medoid")
    ax = fig.add_subplot(gs[1])
    pos = [i["positive_share"] for i in info]
    ax.bar(range(3), [1 - p for p in pos], color=NEG, label="Negative")
    ax.bar(range(3), pos, bottom=[1 - p for p in pos], color=POS, label="Positive")
    for i, p in enumerate(pos):
        ax.text(i, 0.5, f"{p:.0%} pos", ha="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xticks(range(3), [f"C{i}" for i in range(3)]); ax.set_ylim(0, 1); ax.set_title("Sentiment mix")
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    ax.set_title("Most distinctive terms (TF-IDF vs corpus mean;\nHTML artefact 'br' hidden)")
    y = 0.95
    for i in info:
        ax.text(0, y, f"Cluster {i['cluster']}", fontweight="bold", color=CLUSTER_COLORS[i["cluster"]], fontsize=10, va="top")
        ax.text(0, y - 0.09, ", ".join([t for t in i["top_terms"] if t != "br"][:7]), fontsize=8.5, va="top", wrap=True)
        y -= 0.32
    fig.tight_layout()
    out = R.fig / f"kmedoids_profile_n{n}.png"
    fig.savefig(out); plt.close(fig); return out


def svd_variance(R):
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = sorted(int(k) for k in R.var)
    for c, n in zip(plt.cm.viridis(np.linspace(0, 0.9, len(ns))), ns):
        v = np.array(R.var[str(n)])
        ax.plot(np.arange(1, len(v) + 1), v, color=c, label=f"n={n:,} → {v[-1]:.1%}")
    ax.set_xlabel("SVD components"); ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Variance kept by the 50-D SVD input"); ax.legend(fontsize=7)
    out = R.fig / "svd_variance.png"
    fig.savefig(out); plt.close(fig); return out


def make_all(root):
    R = Results(root)
    ns = sorted(R.sel("tsne").n.unique())
    out = [heatmaps(R), metric_lines(R), runtime(R), svd_variance(R), kmedoids_overview(R), sne_quality(R), size_grid(R)]
    for n in (250, 2500, ns[-1]):
        if n in ns:
            out.append(perplexity_grid(R, n))
    if 1000 in R.sel("sne").n.values:
        out += [sne_vs_tsne(R, 1000), perplexity_grid(R, 1000, "sne")]
    out.append(cluster_profile(R, ns[-1]))
    return out
