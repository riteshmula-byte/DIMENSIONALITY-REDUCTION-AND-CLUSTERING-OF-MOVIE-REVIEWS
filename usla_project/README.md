# USLA project: SNE, t-SNE and K-medoids on IMDB reviews

Sweeps: perplexity 5, 10, 15, 20, 25, 30 and sample size 250, 500, 1000, 2500, 5000, 10000, 25000.
Exact SNE stops at n = 2500 (it needs an n x n matrix); t-SNE and K-medoids run to 25000.

## Layout
    usla/data.py        load -> TF-IDF -> SVD (the one shared pipeline)
    usla/sne.py         exact SNE with per-point perplexity calibration (+ sigma=1 baseline)
    usla/kmedoids.py    exact PAM, computed in distance blocks
    usla/metrics.py     trustworthiness, kNN preservation, label agreement
    usla/plots.py       all figures
    run_experiments.py  runs the whole grid, resumable  (--quick for a smoke test)
    make_figures.py     redraws figures from results/
    results/summary.csv one row per run (method, n, perplexity, metrics, seconds)
    results/clusters/   id, sentiment, cluster, dist_to_medoid per K-medoids run
    results/embeddings/ 2-D maps (.npy, float32)
    results/figures/    PNG figures
    report/             USLA_Report.docx / .pdf (+ build script)

## Run
    pip install -r requirements.txt
    # put labeledTrainData.tsv in this folder (or ./data/)
    python run_experiments.py      # about 1 hour on one CPU core
    python make_figures.py
