from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prestige_common import DEFAULT_MASTER_XLSX, DEFAULT_OUTPUT_DIR, TARGET_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera gráficos académicos a partir del Excel maestro del pipeline de prestigio.")
    parser.add_argument(
        "--master-excel",
        default=str(Path(DEFAULT_OUTPUT_DIR) / "reports" / DEFAULT_MASTER_XLSX),
        help="Ruta al Excel maestro generado por prestige_pipeline.py",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(Path(DEFAULT_OUTPUT_DIR) / "plots"),
        help="Carpeta donde guardar los gráficos.",
    )
    return parser.parse_args()


def tidy_target(name: str) -> str:
    return TARGET_LABELS.get(name, name)


def save_heatmap(corr_df: pd.DataFrame, output_path: Path, family: str) -> None:
    subset = corr_df[corr_df["family"] == family].copy() if "family" in corr_df.columns else corr_df.copy()
    if subset.empty:
        return
    pivot = subset.pivot(index="feature", columns="target", values="spearman_rho")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([tidy_target(c) for c in pivot.columns], rotation=35, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Correlaciones de Spearman ({family})")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            text = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_formula_weights(formula_df: pd.DataFrame, output_path: Path) -> None:
    if formula_df.empty:
        return
    plot_df = formula_df.copy().sort_values("best_value", ascending=False)
    x = np.arange(len(plot_df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, plot_df["w_acad"], width, label="Académico")
    ax.bar(x, plot_df["w_exp"], width, label="Experiencial")
    ax.bar(x + width, plot_df["w_peer"], width, label="Pares")
    ax.set_xticks(x)
    ax.set_xticklabels([tidy_target(t) for t in plot_df["target"]], rotation=25, ha="right")
    ax.set_ylabel("Peso optimizado")
    ax.set_ylim(0, 1)
    ax.set_title("Pesos de la fórmula de prestigio optimizados con Optuna")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outer_cv_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    if metrics_df.empty:
        return
    mean_df = metrics_df.groupby("target", as_index=False)[["r2", "mae", "rmse", "spearman_rho"]].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(mean_df))
    width = 0.2
    ax.bar(x - 1.5 * width, mean_df["r2"], width, label="R²")
    ax.bar(x - 0.5 * width, mean_df["mae"], width, label="MAE")
    ax.bar(x + 0.5 * width, mean_df["rmse"], width, label="RMSE")
    ax.bar(x + 1.5 * width, mean_df["spearman_rho"], width, label="Spearman")
    ax.set_xticks(x)
    ax.set_xticklabels([tidy_target(t) for t in mean_df["target"]], rotation=25, ha="right")
    ax.set_title("Desempeño promedio en validación externa por grupos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(perm_df: pd.DataFrame, output_path: Path) -> None:
    if perm_df.empty:
        return
    score = (
        perm_df.groupby("original_feature", as_index=False)["permutation_importance_mean"]
        .mean()
        .sort_values("permutation_importance_mean", ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(score["original_feature"][::-1], score["permutation_importance_mean"][::-1])
    ax.set_title("Importancia de variables (permutation importance final)")
    ax.set_xlabel("Importancia media")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    master_excel = Path(args.master_excel)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    corr_abs = pd.read_excel(master_excel, sheet_name="06_CORR_ABSOLUTE")
    corr_rel = pd.read_excel(master_excel, sheet_name="07_CORR_RELATIVE")
    formula_best = pd.read_excel(master_excel, sheet_name="08_FORMULA_BEST")
    model_outer = pd.read_excel(master_excel, sheet_name="12_MODEL_OUTER_CV")
    perm_agg = pd.read_excel(master_excel, sheet_name="16_FINAL_PERM_AGG")

    if not corr_abs.empty:
        corr_abs["family"] = "absolute"
    if not corr_rel.empty:
        corr_rel["family"] = "relative"
    corr_all = pd.concat([corr_abs, corr_rel], ignore_index=True)

    save_heatmap(corr_all, plots_dir / "correlation_heatmap_absolute.png", family="absolute")
    save_heatmap(corr_all, plots_dir / "correlation_heatmap_relative.png", family="relative")
    save_formula_weights(formula_best, plots_dir / "optuna_formula_weights.png")
    save_outer_cv_metrics(model_outer, plots_dir / "outer_cv_metrics.png")
    save_feature_importance(perm_agg, plots_dir / "final_feature_importance.png")

    print(f"Gráficos guardados en: {plots_dir}")


if __name__ == "__main__":
    main()
