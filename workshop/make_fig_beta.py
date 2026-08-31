"""Build the beta dose-response figure from the completed sweep summaries.

Reads comprehensive_histogram_analysis/*/comprehensive_summary_*.json and plots
live-feature fraction and activation scale against the KL coefficient, with all
other hyperparameters (gelu-1l, d=2048, k=256, auxk=1/32, lr=8e-4, fixed variance)
held constant across the four points.
"""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP_TAG = "gelu-1l_d2048_k256"

pts = []
for f in glob.glob("comprehensive_histogram_analysis/*/comprehensive_summary_*.json"):
    d = json.load(open(f))
    mi, fu, st = d["model_info"], d["feature_usage_summary"], d["overall_activation_stats"]
    if mi.get("dictionary_type") != "VSAETopK":
        continue
    if SWEEP_TAG not in os.path.basename(os.path.dirname(f)):
        continue
    pts.append((mi["kl_coeff"], fu["features_used"], fu["total_features"],
                st["mean"], st["max"]))
pts.sort()

beta   = [p[0] for p in pts]
alive  = [100.0 * p[1] / p[2] for p in pts]
actmn  = [p[3] for p in pts]
actmx  = [p[4] for p in pts]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

ax = axes[0]
ax.plot(beta, alive, "o-", color="#B3202C", lw=1.8, ms=6)
ax.set_xscale("log")
ax.set_xlabel(r"KL coefficient $\beta$")
ax.set_ylabel(r"live features (\% of 2048)" if plt.rcParams["text.usetex"] else "live features (% of 2048)")
ax.set_title("Feature death is monotonic in " + r"$\beta$", fontsize=10)
ax.grid(alpha=0.3, lw=0.6)
ax.set_ylim(0, 100)
for b, a in zip(beta, alive):
    ax.annotate(f"{a:.0f}%", (b, a), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)

ax = axes[1]
ax.plot(beta, actmn, "o-", color="#1F4E79", lw=1.8, ms=6, label="mean |activation|")
ax.plot(beta, actmx, "s--", color="#7F7F7F", lw=1.4, ms=5, label="max activation")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"KL coefficient $\beta$")
ax.set_ylabel("activation magnitude")
ax.set_title("Activation scale collapses with " + r"$\beta$", fontsize=10)
ax.grid(alpha=0.3, lw=0.6, which="both")
ax.legend(fontsize=8, frameon=False)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
# Omit the PDF CreationDate so regenerating the figure is byte-reproducible.
# Without this, matplotlib stamps the current time into the PDF and the file
# shows as modified on every run even when the plot is identical.
fig.savefig(
    "workshop/figs/beta_sweep.pdf",
    bbox_inches="tight",
    metadata={"CreationDate": None},
)
fig.savefig("workshop/figs/beta_sweep.png", dpi=180, bbox_inches="tight")

print(f"{'beta':>10} {'live':>6} {'%alive':>8} {'mean_act':>9} {'max_act':>8}")
for (b, u, t, mn, mx), a in zip(pts, alive):
    print(f"{b:>10} {u:>6} {a:>7.1f}% {mn:>9.3f} {mx:>8.2f}")
