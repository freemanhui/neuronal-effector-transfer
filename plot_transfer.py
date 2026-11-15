"""Recreate main Figure 2 from the published target-resampling interval table."""
import csv
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
PAIRS = [("StAnge","StAnge"),("Iida","Iida"),("Russ","Russ"),("StAnge","Gao"),("Iida","Russ"),("Russ","Iida"),
         ("StAnge","Iida"),("StAnge","Russ"),("Iida","StAnge"),("Iida","Gao"),("Russ","StAnge"),("Russ","Gao")]


def plot(table, output):
    with table.open() as f:
        rows = {(r["source"],r["target"],r["arm"],r["metric"]):r for r in csv.DictReader(f, delimiter="\t") if r["model"] == "TF"}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 4.35), sharey=True)
    for ax, metric, label in zip(axes, ("rho","amplitude","R2_centered"), ("Signed alignment ρ","Amplitude ratio a","Centred R²")):
        ax.axvspan(0, 0, color="white")
        ax.axhspan(2.5, 5.5, color="#edf2f3", zorder=0)
        ax.axvline(0, color="#657079", lw=.8, ls=":")
        for i, (source, target) in enumerate(PAIRS):
            for arm, delta, color, marker in (("raw",-.13,"#263853","o"),("X_calibrated",.13,"#087a78","s")):
                r = rows[source,target,arm,metric]
                low, point, high = map(float,(r["ci025"],r["point"],r["ci975"]))
                ax.plot([low,high],[i+delta]*2, color=color, lw=1.15, solid_capstyle="round")
                ax.plot(point,i+delta,marker, ms=3.5,color=color, markerfacecolor=color if arm=="raw" else "white",markeredgewidth=.9)
        ax.set_title(label, fontsize=10, fontweight="bold",loc="left", pad=7)
        for edge in ("top","right","left"): ax.spines[edge].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d8dedf", lw=.45)
        ax.set_ylim(11.6,-.6)
    axes[0].set_yticks(range(12),[f"{s} → {t}" for s,t in PAIRS],fontsize=8)
    for ax in axes[1:]: ax.tick_params(labelleft=False)
    handles = [Line2D([],[],color=c,marker=m,lw=1,ms=4,markerfacecolor=fill,label=lbl)
               for c,m,fill,lbl in (("#263853","o","#263853","Source scaling"),("#087a78","s","white","Target-X scaling"))]
    fig.legend(handles=handles,loc="lower center",ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.165,right=.995,bottom=.15,top=.92,wspace=.17)
    fig.savefig(output,metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("intervals",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    plot(args.intervals,args.output)
