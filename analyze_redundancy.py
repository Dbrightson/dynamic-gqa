#!/usr/bin/env python3
"""Stage 2b: Offline analysis of head redundancy (CPU only).

Reads stats_<domain>.npz from collect_activations.py and produces:
  results/analysis/summary.txt      human-readable verdicts
  results/analysis/redundancy.csv   rho(tau) per layer per domain
  results/analysis/ari.csv          cross-domain adjusted Rand index per layer
  results/analysis/clusters.json    per-domain, per-layer KV-head groupings
                                    for G in {2,4} (input to simulate_sharing)
  results/analysis/*.png            similarity heatmaps
"""
import argparse, glob, json, os
import numpy as np

def cluster(sim, G):
    from sklearn.cluster import AgglomerativeClustering
    G = min(G, sim.shape[0])   # cannot have more clusters than heads
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    return AgglomerativeClustering(
        n_clusters=G, metric="precomputed", linkage="average"
    ).fit_predict(dist).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="results/activations")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--taus", default="0.80,0.90,0.95,0.99")
    ap.add_argument("--groups", default="2,4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    taus = [float(t) for t in args.taus.split(",")]
    Gs = [int(g) for g in args.groups.split(",")]

    from sklearn.metrics import adjusted_rand_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = {}
    for f in sorted(glob.glob(os.path.join(args.acts, "stats_*.npz"))):
        dom = os.path.basename(f)[len("stats_"):-len(".npz")]
        stats[dom] = np.load(f)
    if not stats:
        raise SystemExit(f"No stats_*.npz found in {args.acts}")
    doms = list(stats)
    L, Hkv, _ = stats[doms[0]]["k_cos"].shape
    print(f"[analyze] domains={doms} L={L} Hkv={Hkv}")

    # --- redundancy profiles rho_l(tau) --------------------------------
    off = ~np.eye(Hkv, dtype=bool)
    rows = ["domain,layer," + ",".join(f"rho@{t}" for t in taus)]
    frac_layers_redundant = {}
    for dom in doms:
        kc = stats[dom]["k_cos"]
        hit = 0
        for l in range(L):
            vals = kc[l][off]
            rho = [float((vals > t).mean()) for t in taus]
            rows.append(f"{dom},{l}," + ",".join(f"{r:.4f}" for r in rho))
            if rho[taus.index(0.90)] >= 0.30 if 0.90 in taus else False:
                hit += 1
        frac_layers_redundant[dom] = hit / L
    open(os.path.join(args.out, "redundancy.csv"), "w").write("\n".join(rows))

    # --- cross-domain cluster stability (ARI) ---------------------------
    labels = {dom: {G: [cluster(stats[dom]["k_cos"][l], G) for l in range(L)]
                    for G in Gs} for dom in doms}
    ari_rows = ["G,layer,dom_a,dom_b,ari"]
    mean_ari = {}
    for G in Gs:
        aris = []
        for l in range(L):
            for i, a in enumerate(doms):
                for b in doms[i+1:]:
                    s = adjusted_rand_score(labels[a][G][l], labels[b][G][l])
                    ari_rows.append(f"{G},{l},{a},{b},{s:.4f}")
                    aris.append(s)
        mean_ari[G] = float(np.mean(aris)) if aris else float("nan")
    open(os.path.join(args.out, "ari.csv"), "w").write("\n".join(ari_rows))

    json.dump({"domains": doms, "groups": Gs, "labels": labels},
              open(os.path.join(args.out, "clusters.json"), "w"), indent=1)

    # --- plots -----------------------------------------------------------
    for dom in doms:
        kc = stats[dom]["k_cos"]
        fig, ax = plt.subplots(figsize=(6, 3))
        m = np.array([kc[l][off].mean() for l in range(L)])
        ax.plot(m); ax.set_xlabel("layer"); ax.set_ylabel("mean off-diag K-cos")
        ax.set_title(f"K-head similarity by layer ({dom})")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"kcos_{dom}.png"), dpi=120)
        plt.close(fig)

    # --- verdicts (pre-registered criteria from the paper, Sec. 6) ------
    lines = []
    lines.append("=== Stage 2 verdicts (pre-registered) ===")
    for dom, frac in frac_layers_redundant.items():
        lines.append(f"[{dom}] fraction of layers with rho(0.90) >= 0.30: {frac:.2f} "
                     f"({'PASS' if frac >= 0.25 else 'fail'} vs >=0.25)")
    for G, a in mean_ari.items():
        lines.append(f"[G={G}] mean cross-domain ARI: {a:.3f} "
                     f"({'PASS (input-dependent clusters)' if a < 0.5 else 'fail (clusters are static)'} vs <0.5)")
    lines.append("")
    lines.append("Interpretation: PASS on both => evidence for Dynamic GQA;")
    lines.append("high redundancy + high ARI => argues for better *static* grouping instead.")
    txt = "\n".join(lines)
    open(os.path.join(args.out, "summary.txt"), "w").write(txt)
    print(txt)

if __name__ == "__main__":
    main()
