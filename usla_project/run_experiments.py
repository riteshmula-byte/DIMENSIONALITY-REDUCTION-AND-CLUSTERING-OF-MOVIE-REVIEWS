"""Run every experiment once and write tidy results to ./results.

    python run_experiments.py            # full grid (resumable)
    python run_experiments.py --quick    # tiny grid for a smoke test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from usla.data import build_features, find_data, load_reviews
from usla.kmedoids import k_medoids
from usla.metrics import neighbourhood_metrics
from usla.sne import fit_sne

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
SIZES = [250, 500, 1000, 2500, 5000, 10000, 25000]
PERPLEXITIES = [5, 10, 15, 20, 25, 30]
SNE_MAX_N = 2500
K = 3
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sizes = [250, 500] if args.quick else SIZES
    perps = [5, 30] if args.quick else PERPLEXITIES
    for d in ("embeddings", "clusters"):
        (RES / d).mkdir(parents=True, exist_ok=True)
    summ_path = RES / "summary.csv"
    rows = pd.read_csv(summ_path).to_dict("records") if summ_path.exists() else []
    done = {(r["method"], int(r["n"]), r["perplexity"], r["k"]) for r in rows}
    terms_path, var_path = RES / "cluster_terms.json", RES / "svd_variance.json"
    terms = json.loads(terms_path.read_text()) if terms_path.exists() else {}
    var = json.loads(var_path.read_text()) if var_path.exists() else {}

    def key(m, n, p=None, k=None):
        return (m, n, np.nan if p is None else float(p), np.nan if k is None else float(k))

    def seen(m, n, p=None, k=None):
        kk = key(m, n, p, k)
        return any(kk[0] == d[0] and kk[1] == d[1] and
                   (kk[2] == d[2] or (np.isnan(kk[2]) and np.isnan(d[2]))) and
                   (kk[3] == d[3] or (np.isnan(kk[3]) and np.isnan(d[3]))) for d in done)

    def save():
        pd.DataFrame(rows).to_csv(summ_path, index=False)
        terms_path.write_text(json.dumps(terms))
        var_path.write_text(json.dumps(var))

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)

    reviews = load_reviews(find_data(ROOT))
    for n in sizes:
        f = build_features(reviews, n, seed=SEED)
        var[str(n)] = f.cum_variance.tolist()
        base = {"n": n, "svd_variance": float(f.cum_variance[-1]),
                "positive_share": float(f.labels.mean())}

        for k in ([K, 2, 4, 5, 6, 7, 8] if n == 2500 else [K]):
            if seen("kmedoids", n, None, k):
                continue
            t = time.time()
            r = k_medoids(f.Z, k)
            sec = time.time() - t
            sil = silhouette_score(f.Z, r["labels"], sample_size=min(n, 5000), random_state=SEED)
            row = dict(base, method="kmedoids", perplexity=np.nan, k=k, seconds=sec,
                       cost=r["cost"], cost_per_point=r["cost"] / n, swaps=r["swaps"],
                       silhouette=sil, ari_vs_sentiment=adjusted_rand_score(f.labels, r["labels"]),
                       nmi_vs_sentiment=normalized_mutual_info_score(f.labels, r["labels"]),
                       cluster_sizes=json.dumps(np.bincount(r["labels"], minlength=k).tolist()))
            rows.append(row); done.add(key("kmedoids", n, None, k))
            pd.DataFrame({"id": f.ids, "sentiment": f.labels, "cluster": r["labels"],
                          "dist_to_medoid": r["dist_to_medoid"].round(5)}).to_csv(
                RES / "clusters" / f"kmedoids_n{n}_k{k}.csv", index=False)
            if k == K:
                X = f.tfidf.tocsr()
                overall = np.asarray(X.mean(axis=0)).ravel()
                terms[str(n)] = [{
                    "cluster": c, "size": int((r["labels"] == c).sum()),
                    "positive_share": float(f.labels[r["labels"] == c].mean()),
                    "medoid_id": str(f.ids[r["medoids"][c]]),
                    "top_terms": f.terms[np.argsort(-(np.asarray(
                        X[r["labels"] == c].mean(axis=0)).ravel() - overall))[:8]].tolist()}
                    for c in range(k)]
            save(); log("kmedoids", n, k, f"{sec:.1f}s cost/pt={r['cost']/n:.4f}")

        for p in perps:
            if seen("tsne", n, p) or p >= n - 1:
                continue
            t = time.time()
            ts = TSNE(2, perplexity=p, init="pca", learning_rate="auto", random_state=SEED)
            Y = ts.fit_transform(f.Z)
            sec = time.time() - t
            m = neighbourhood_metrics(f.Z, Y, f.labels)
            rows.append(dict(base, method="tsne", perplexity=p, k=np.nan, seconds=sec,
                             kl=float(ts.kl_divergence_), iterations=int(ts.n_iter_), **m))
            done.add(key("tsne", n, p))
            np.save(RES / "embeddings" / f"tsne_n{n}_p{p}.npy", Y.astype(np.float32))
            save(); log("tsne", n, p, f"{sec:.1f}s KL={ts.kl_divergence_:.3f} trust={m['trustworthiness']:.3f}")

        if n <= SNE_MAX_N:
            for p in perps + ["sigma1"]:
                pp = np.nan if p == "sigma1" else p
                tag = "sigma1" if p == "sigma1" else f"p{p}"
                if seen("sne" if p != "sigma1" else "sne_sigma1", n, pp):
                    continue
                r = fit_sne(f.Z, **({"sigma": 1.0} if p == "sigma1" else {"perplexity": p}), seed=SEED)
                m = neighbourhood_metrics(f.Z, r["Y"], f.labels)
                meth = "sne_sigma1" if p == "sigma1" else "sne"
                rows.append(dict(base, method=meth, perplexity=pp, k=np.nan, seconds=r["seconds"],
                                 kl=r["loss"] / n, iterations=r["nit"],
                                 eff_perplexity=r["eff_perplexity"], **m))
                done.add(key(meth, n, pp))
                np.save(RES / "embeddings" / f"sne_n{n}_{tag}.npy", r["Y"].astype(np.float32))
                save(); log("sne", n, tag, f"{r['seconds']:.1f}s KL/pt={r['loss']/n:.3f} trust={m['trustworthiness']:.3f}")
    save(); log("done")


if __name__ == "__main__":
    main()
