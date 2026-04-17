from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from prestige_common import DEFAULT_OUTPUT_DIR, TARGET_LABELS, EXPECTED_COLUMNS, load_data_bundle, load_model_bundle, safe_spearman


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcula y exporta explicabilidad SHAP para los modelos finales del pipeline de prestigio.")
    parser.add_argument(
        "--input",
        default="consolidado_ordenado.xlsx",
        help="Ruta al Excel consolidado original.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(Path(DEFAULT_OUTPUT_DIR) / "models"),
        help="Carpeta con los modelos .joblib generados por prestige_pipeline.py",
    )
    parser.add_argument(
        "--shap-dir",
        default=str(Path(DEFAULT_OUTPUT_DIR) / "shap"),
        help="Carpeta de salida para tablas y figuras SHAP.",
    )
    return parser.parse_args()


def tidy_target(name: str) -> str:
    return TARGET_LABELS.get(name, name)


def explanation_for_model(bundle) -> shap.Explanation:
    preprocessor = bundle.pipeline.named_steps["preprocess"]
    model = bundle.pipeline.named_steps["model"]
    X_trans = preprocessor.transform(bundle.X_train_full)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    expected_value = explainer.expected_value
    base_values = np.repeat(expected_value if np.isscalar(expected_value) else expected_value[0], X_trans.shape[0])
    return shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_trans,
        feature_names=bundle.transformed_feature_names,
    )


def global_tables(bundle, explanation: shap.Explanation) -> Dict[str, pd.DataFrame]:
    feature_names = list(explanation.feature_names)
    values = np.asarray(explanation.values)
    data = np.asarray(explanation.data)
    original_map = bundle.transformed_to_original

    transformed_df = pd.DataFrame(
        {
            "target": bundle.target,
            "transformed_feature": feature_names,
            "original_feature": [original_map.get(name, name) for name in feature_names],
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    agg_df = (
        transformed_df.groupby(["target", "original_feature"], as_index=False)[["mean_abs_shap", "mean_shap"]]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    direction_rows: List[Dict[str, object]] = []
    for idx, feature_name in enumerate(feature_names):
        original_feature = original_map.get(feature_name, feature_name)
        feature_values = pd.Series(data[:, idx])
        shap_series = pd.Series(values[:, idx])
        rho, p_value, n_obs = safe_spearman(feature_values, shap_series)
        direction = "positive" if pd.notna(rho) and rho > 0 else "negative" if pd.notna(rho) and rho < 0 else "unclear"
        direction_rows.append(
            {
                "target": bundle.target,
                "transformed_feature": feature_name,
                "original_feature": original_feature,
                "direction_hint": direction,
                "spearman_feature_vs_shap": rho,
                "p_value": p_value,
                "n": n_obs,
            }
        )
    direction_df = pd.DataFrame(direction_rows).sort_values("spearman_feature_vs_shap", ascending=False)

    long_df = pd.DataFrame(values, columns=feature_names)
    long_df.insert(0, "row_id", np.arange(len(long_df)))
    long_df.insert(1, "target", bundle.target)
    long_melt = long_df.melt(id_vars=["row_id", "target"], var_name="transformed_feature", value_name="shap_value")

    data_df = pd.DataFrame(data, columns=feature_names)
    data_df.insert(0, "row_id", np.arange(len(data_df)))
    data_melt = data_df.melt(id_vars=["row_id"], var_name="transformed_feature", value_name="feature_value")
    long_melt = long_melt.merge(data_melt, on=["row_id", "transformed_feature"], how="left")
    long_melt["original_feature"] = long_melt["transformed_feature"].map(original_map)

    return {
        "global_transformed": transformed_df,
        "global_aggregated": agg_df,
        "direction": direction_df,
        "long": long_melt,
    }


def local_cases_table(bundle, explanation: shap.Explanation) -> pd.DataFrame:
    y = bundle.y_train_full.reset_index(drop=True)
    order = y.sort_values().index.to_list()
    if not order:
        return pd.DataFrame()
    case_indices = {
        "lowest_target": order[0],
        "median_target": order[len(order) // 2],
        "highest_target": order[-1],
    }

    rows: List[Dict[str, object]] = []
    feature_names = list(explanation.feature_names)
    values = np.asarray(explanation.values)
    for label, idx in case_indices.items():
        contributions = pd.DataFrame(
            {
                "transformed_feature": feature_names,
                "original_feature": [bundle.transformed_to_original.get(name, name) for name in feature_names],
                "shap_value": values[idx],
            }
        )
        contributions = contributions.reindex(contributions["shap_value"].abs().sort_values(ascending=False).index).head(12)
        contributions.insert(0, "case_label", label)
        contributions.insert(1, "target", bundle.target)
        contributions.insert(2, "row_id", int(idx))
        contributions.insert(3, "observed_target", float(y.iloc[idx]))
        rows.append(contributions)
    return pd.concat(rows, ignore_index=True)


def save_summary_plot(explanation: shap.Explanation, output_path: Path, target: str) -> None:
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=12, show=False)
    plt.title(f"SHAP summary plot: {tidy_target(target)}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_bar_plot(global_df: pd.DataFrame, output_path: Path, target: str) -> None:
    top = global_df.head(12).copy().sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["original_feature"], top["mean_abs_shap"])
    ax.set_title(f"Importancia global SHAP: {tidy_target(target)}")
    ax.set_xlabel("mean(|SHAP|)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dependence_plots(bundle, explanation: shap.Explanation, output_dir: Path) -> None:
    numeric_candidates = [f"num__{name}" for name in bundle.numeric_features]
    importance = pd.DataFrame(
        {
            "feature": list(explanation.feature_names),
            "mean_abs_shap": np.abs(np.asarray(explanation.values)).mean(axis=0),
        }
    )
    chosen = importance[importance["feature"].isin(numeric_candidates)].sort_values("mean_abs_shap", ascending=False).head(3)
    for _, row in chosen.iterrows():
        feat = row["feature"]
        plt.figure()
        shap.plots.scatter(explanation[:, feat], show=False)
        plt.title(f"Dependence plot: {tidy_target(bundle.target)} | {feat}")
        plt.tight_layout()
        plt.savefig(output_dir / f"dependence_{bundle.target}_{feat.replace('/', '_')}.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_waterfall_plots(bundle, explanation: shap.Explanation, output_dir: Path) -> None:
    y = bundle.y_train_full.reset_index(drop=True)
    order = y.sort_values().index.to_list()
    if not order:
        return
    case_indices = {
        "lowest": order[0],
        "median": order[len(order) // 2],
        "highest": order[-1],
    }
    for label, idx in case_indices.items():
        plt.figure()
        shap.plots.waterfall(explanation[idx], max_display=12, show=False)
        plt.title(f"Waterfall SHAP: {tidy_target(bundle.target)} | {label}")
        plt.tight_layout()
        plt.savefig(output_dir / f"waterfall_{bundle.target}_{label}.png", dpi=300, bbox_inches="tight")
        plt.close()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    shap_dir = Path(args.shap_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)

    summary_frames: List[pd.DataFrame] = []
    agg_frames: List[pd.DataFrame] = []
    direction_frames: List[pd.DataFrame] = []
    long_frames: List[pd.DataFrame] = []
    local_frames: List[pd.DataFrame] = []

    for model_path in sorted(models_dir.glob("model_*.joblib")):
        bundle = load_model_bundle(model_path)
        explanation = explanation_for_model(bundle)
        tables = global_tables(bundle, explanation)
        local_df = local_cases_table(bundle, explanation)

        save_summary_plot(explanation, shap_dir / f"summary_{bundle.target}.png", bundle.target)
        save_bar_plot(tables["global_aggregated"], shap_dir / f"bar_{bundle.target}.png", bundle.target)
        save_dependence_plots(bundle, explanation, shap_dir)
        save_waterfall_plots(bundle, explanation, shap_dir)

        summary_frames.append(tables["global_transformed"])
        agg_frames.append(tables["global_aggregated"])
        direction_frames.append(tables["direction"])
        long_frames.append(tables["long"])
        if not local_df.empty:
            local_frames.append(local_df)

    global_transformed_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    global_aggregated_df = pd.concat(agg_frames, ignore_index=True) if agg_frames else pd.DataFrame()
    direction_df = pd.concat(direction_frames, ignore_index=True) if direction_frames else pd.DataFrame()
    long_df = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    local_df = pd.concat(local_frames, ignore_index=True) if local_frames else pd.DataFrame()

    summary_excel = shap_dir / "shap_outputs.xlsx"
    with pd.ExcelWriter(summary_excel, engine="openpyxl") as writer:
        global_transformed_df.to_excel(writer, sheet_name="01_GLOBAL_TRANSFORMED", index=False)
        global_aggregated_df.to_excel(writer, sheet_name="02_GLOBAL_AGGREGATED", index=False)
        direction_df.to_excel(writer, sheet_name="03_DIRECTION_HINTS", index=False)
        local_df.to_excel(writer, sheet_name="04_LOCAL_CASES", index=False)
        long_df.to_excel(writer, sheet_name="05_LONG_VALUES", index=False)

    print(f"Outputs SHAP guardados en: {shap_dir}")
    print(f"Resumen Excel SHAP: {summary_excel}")


if __name__ == "__main__":
    main()
