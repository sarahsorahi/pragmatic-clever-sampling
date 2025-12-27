import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================

BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/code_scripts")

CONDITIONS = {
    "Real-only": BASE_DIR / "results_real_only" / "results_real_only_runs.csv",
    "Real + all synthetic": BASE_DIR / "results_real_all_synthetic_" / "results_real_all_synthetic_runs.csv",
    "Core-anchored": BASE_DIR / "results_core_anchored_" / "results_core_anchored_runs.csv",
    "Boundary-sampled": BASE_DIR / "results_boundary_sampled" / "results_boundary_sampled_runs.csv",
}

OUT_DIR = BASE_DIR / "final_comparison"
OUT_DIR.mkdir(exist_ok=True)

LABELS = ["INTJ", "DM", "DIR"]

# =========================
# Load + aggregate
# =========================

rows = []

for condition, path in CONDITIONS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    row = {
        "condition": condition,
        "macro_f1_mean": df["macro_f1"].mean(),
        "macro_f1_std": df["macro_f1"].std(),
    }

    for lbl in LABELS:
        row[f"f1_{lbl}_mean"] = df[f"f1_{lbl}"].mean()
        row[f"f1_{lbl}_std"] = df[f"f1_{lbl}"].std()

    rows.append(row)

summary_df = pd.DataFrame(rows)

# Δ vs real-only
real_macro = summary_df.loc[
    summary_df["condition"] == "Real-only", "macro_f1_mean"
].values[0]

summary_df["delta_vs_real_only"] = (
    summary_df["macro_f1_mean"] - real_macro
)

# =========================
# Save outputs
# =========================

summary_df.to_csv(
    OUT_DIR / "comparison_table.csv",
    index=False
)

with open(OUT_DIR / "comparison_summary.txt", "w") as f:
    f.write("Comparison of sampling conditions (mean ± std)\n\n")
    for _, r in summary_df.iterrows():
        f.write(f"{r['condition']}:\n")
        f.write(f"  Macro F1 = {r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f}\n")
        for lbl in LABELS:
            f.write(
                f"  F1 {lbl} = "
                f"{r[f'f1_{lbl}_mean']:.3f} ± {r[f'f1_{lbl}_std']:.3f}\n"
            )
        f.write(f"  Δ vs Real-only = {r['delta_vs_real_only']:+.3f}\n\n")

print(" Final comparison completed.")
print(f"Saved to: {OUT_DIR}")
