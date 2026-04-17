from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy.stats import spearmanr

from prestige_common import (
    DEFAULT_INPUT,
    DEFAULT_MASTER_XLSX,
    DEFAULT_OUTPUT_DIR,
    EXPECTED_COLUMNS,
    MAIN_TARGETS,
    SEED,
    DataBundle,
    ModelBundle,
    aggregate_permutation_importance,
    build_pipeline,
    build_readme_sheet,
    get_correlation_tables,
    get_feature_spec,
    group_kfold_splits,
    json_dump,
    load_data_bundle,
    precompute_feature_name_mapping,
    regression_metrics,
    save_model_bundle,
    safe_spearman,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstrucción robusta del pipeline de prestigio con validación por grupos, Optuna y outputs auditables.",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Ruta al Excel consolidado.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Carpeta base de salida.")
    parser.add_argument("--model-trials", type=int, default=30, help="Número de trials Optuna para modelos.")
    parser.add_argument("--formula-trials", type=int, default=80, help="Número de trials Optuna para pesos de fórmula.")
    parser.add_argument(
        "--outer-splits",
        type=int,
        default=5,
        help="Número máximo de folds externos para validación por grupos.",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=4,
        help="Número máximo de folds internos para optimización bayesiana.",
    )
    return parser.parse_args()


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    from prestige_common import ensure_output_dirs as _ensure_output_dirs

    return _ensure_output_dirs(base_dir)


def normalize_formula_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in raw_weights.values())
    if math.isclose(total, 0.0):
        return {k: 1.0 / len(raw_weights) for k in raw_weights}
    return {k: max(v, 0.0) / total for k, v in raw_weights.items()}

def scale_or_keep_from_train(train_num: pd.Series, test_num: pd.Series):
    train_num = pd.to_numeric(train_num, errors="coerce")
    test_num = pd.to_numeric(test_num, errors="coerce")

    valid = train_num.dropna()
    if valid.empty:
        return (
            pd.Series(np.nan, index=train_num.index, dtype="float64"),
            pd.Series(np.nan, index=test_num.index, dtype="float64"),
            "empty_train",
        )

    if ((valid >= 0) & (valid <= 1)).all():
        return train_num.astype("float64"), test_num.astype("float64"), "kept_as_is"

    min_val = float(valid.min())
    max_val = float(valid.max())
    if math.isclose(min_val, max_val):
        return (
            pd.Series(0.0, index=train_num.index),
            pd.Series(0.0, index=test_num.index),
            "constant_zero",
        )

    train_scaled = (train_num - min_val) / (max_val - min_val)
    test_scaled = (test_num - min_val) / (max_val - min_val)
    return train_scaled, test_scaled, "minmax_rescaled"

def fold_scaled_components(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    component_map = {
        "acad": EXPECTED_COLUMNS["acad"],
        "peer": EXPECTED_COLUMNS["peer"],
        "exp_proj": EXPECTED_COLUMNS["exp_proj"],
        "exp_plan": EXPECTED_COLUMNS["exp_plan"],
        "exp_model": EXPECTED_COLUMNS["exp_model"],
    }
    train = train_df.copy()
    test = test_df.copy()

    for key in ["acad", "peer", "exp_proj", "exp_plan", "exp_model"]:
        col = component_map[key]
        train_scaled, test_scaled, _ = scale_or_keep_from_train(train[col], test[col])
        train[f"__{key}_01"] = train_scaled
        test[f"__{key}_01"] = test_scaled

    train["__exp_01"] = train[["__exp_proj_01", "__exp_plan_01", "__exp_model_01"]].mean(axis=1)
    test["__exp_01"] = test[["__exp_proj_01", "__exp_plan_01", "__exp_model_01"]].mean(axis=1)

    train["__prestige_formula_input_acad"] = train["__acad_01"]
    train["__prestige_formula_input_peer"] = train["__peer_01"]
    train["__prestige_formula_input_exp"] = train["__exp_01"]

    test["__prestige_formula_input_acad"] = test["__acad_01"]
    test["__prestige_formula_input_peer"] = test["__peer_01"]
    test["__prestige_formula_input_exp"] = test["__exp_01"]

    return train, test


def formula_objective_factory(df: pd.DataFrame, target: str, n_splits: int):
    group_col = EXPECTED_COLUMNS["group"]

    def objective(trial: optuna.Trial) -> float:
        raw_weights = {
            "acad": trial.suggest_float("w_acad_raw", 0.0, 1.0),
            "exp": trial.suggest_float("w_exp_raw", 0.0, 1.0),
            "peer": trial.suggest_float("w_peer_raw", 0.0, 1.0),
        }
        weights = normalize_formula_weights(raw_weights)
        folds = []
        gkf = group_kfold_splits(df[group_col], n_splits=n_splits)
        for train_idx, test_idx in gkf.split(df, df[target], groups=df[group_col]):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            train_df, test_df = fold_scaled_components(train_df, test_df)
            formula_values = (
                weights["acad"] * test_df["__prestige_formula_input_acad"]
                + weights["exp"] * test_df["__prestige_formula_input_exp"]
                + weights["peer"] * test_df["__prestige_formula_input_peer"]
            )
            rho, _, _ = safe_spearman(formula_values, test_df[target])
            folds.append(rho)
        mean_rho = float(np.nanmean(folds)) if folds else np.nan
        trial.set_user_attr("mean_test_spearman", mean_rho)
        trial.set_user_attr("weights", weights)
        return -999.0 if pd.isna(mean_rho) else mean_rho

    return objective


def optimize_formula_weights(df: pd.DataFrame, target: str, n_trials: int, n_splits: int) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(formula_objective_factory(df, target, n_splits=n_splits), n_trials=n_trials, show_progress_bar=False)

    trials_rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        weights = trial.user_attrs.get("weights", {})
        trials_rows.append(
            {
                "target": target,
                "trial_number": trial.number,
                "objective_value": trial.value,
                "w_acad": weights.get("acad"),
                "w_exp": weights.get("exp"),
                "w_peer": weights.get("peer"),
            }
        )

    best_weights = study.best_trial.user_attrs["weights"]
    gkf = group_kfold_splits(df[EXPECTED_COLUMNS["group"]], n_splits=n_splits)
    fold_rows: List[Dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(df, df[target], groups=df[EXPECTED_COLUMNS["group"]]), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_df, test_df = fold_scaled_components(train_df, test_df)
        formula_values = (
            best_weights["acad"] * test_df["__prestige_formula_input_acad"]
            + best_weights["exp"] * test_df["__prestige_formula_input_exp"]
            + best_weights["peer"] * test_df["__prestige_formula_input_peer"]
        )
        rho, p_value, n_obs = safe_spearman(formula_values, test_df[target])
        fold_rows.append(
            {
                "target": target,
                "fold": fold_id,
                "test_groups": int(test_df[EXPECTED_COLUMNS["group"]].nunique()),
                "test_rows": int(len(test_df)),
                "spearman_rho": rho,
                "p_value": p_value,
                "n": n_obs,
                "w_acad": best_weights["acad"],
                "w_exp": best_weights["exp"],
                "w_peer": best_weights["peer"],
            }
        )

    target_values = (
        best_weights["acad"] * df["Prestige_Academic_01"]
        + best_weights["exp"] * df["Prestige_Experiential_01"]
        + best_weights["peer"] * df["Prestige_Peer_01"]
    )
    best_summary = {
        "target": target,
        "best_value": study.best_value,
        "w_acad": best_weights["acad"],
        "w_exp": best_weights["exp"],
        "w_peer": best_weights["peer"],
        "formula_column": f"Prestige_Formula_Optuna__{target}",
        "n_trials": n_trials,
    }
    df[f"Prestige_Formula_Optuna__{target}"] = target_values
    return best_summary, pd.DataFrame(trials_rows), pd.DataFrame(fold_rows)


def suggest_rf_params(trial: optuna.Trial) -> Dict[str, Any]:
    max_depth_choice = trial.suggest_categorical("max_depth", [None, 3, 4, 5, 6, 8, 10, 12])
    max_features_choice = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75, 1.0])
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": max_depth_choice,
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": max_features_choice,
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
    }


def optuna_model_objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    n_splits: int,
) -> float:
    params = suggest_rf_params(trial)
    gkf = group_kfold_splits(groups, n_splits=n_splits)
    maes: List[float] = []
    rhos: List[float] = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipe = build_pipeline(numeric_features, categorical_features, params)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        maes.append(metrics["mae"])
        rhos.append(metrics["spearman_rho"])
    mean_mae = float(np.mean(maes))
    mean_rho = float(np.nanmean(rhos))
    trial.set_user_attr("mean_mae", mean_mae)
    trial.set_user_attr("mean_spearman", mean_rho)
    return mean_mae


def run_inner_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    n_trials: int,
    n_splits: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: optuna_model_objective(
            trial,
            X_train,
            y_train,
            groups_train,
            numeric_features,
            categorical_features,
            n_splits=n_splits,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    trials_rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": trial.number,
            "objective_mae": trial.value,
            "mean_spearman": trial.user_attrs.get("mean_spearman"),
        }
        row.update(trial.params)
        trials_rows.append(row)

    return study.best_params, pd.DataFrame(trials_rows)


def fit_final_model_bundle(
    derived_df: pd.DataFrame,
    target: str,
    best_params: Dict[str, Any],
) -> ModelBundle:
    feature_columns, numeric_features, categorical_features = get_feature_spec(derived_df)
    model_df = derived_df[[EXPECTED_COLUMNS["group"]] + feature_columns + [target]].copy()
    model_df = model_df.dropna(subset=[target]).reset_index(drop=True)
    X = model_df[feature_columns].copy()
    y = pd.to_numeric(model_df[target], errors="coerce")
    groups = model_df[EXPECTED_COLUMNS["group"]]

    pipe = build_pipeline(numeric_features, categorical_features, best_params)
    pipe.fit(X, y)
    transformed_feature_names, transformed_to_original = precompute_feature_name_mapping(
        pipe.named_steps["preprocess"],
        numeric_features,
        categorical_features,
    )
    return ModelBundle(
        target=target,
        pipeline=pipe,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        transformed_feature_names=transformed_feature_names,
        transformed_to_original=transformed_to_original,
        best_params=best_params,
        X_train_full=X,
        y_train_full=y,
        groups_train_full=groups,
    )


def evaluate_model_nested_cv(
    derived_df: pd.DataFrame,
    target: str,
    n_trials: int,
    outer_splits: int,
    inner_splits: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    feature_columns, numeric_features, categorical_features = get_feature_spec(derived_df)
    model_df = derived_df[[EXPECTED_COLUMNS["id"], EXPECTED_COLUMNS["group"], EXPECTED_COLUMNS["team_member"], *feature_columns, target]].copy()
    model_df = model_df.dropna(subset=[target]).reset_index(drop=True)
    X = model_df[feature_columns]
    y = pd.to_numeric(model_df[target], errors="coerce")
    groups = model_df[EXPECTED_COLUMNS["group"]]

    outer_cv = group_kfold_splits(groups, n_splits=outer_splits)
    metrics_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

        best_params, fold_trials_df = run_inner_optuna(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            n_trials=n_trials,
            n_splits=min(inner_splits, int(groups_train.nunique())),
        )
        if not fold_trials_df.empty:
            fold_trials_df.insert(0, "target", target)
            fold_trials_df.insert(1, "outer_fold", outer_fold)
            trial_rows.append(fold_trials_df)

        pipe = build_pipeline(numeric_features, categorical_features, best_params)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        metrics_rows.append(
            {
                "target": target,
                "outer_fold": outer_fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_groups": int(groups_train.nunique()),
                "test_groups": int(groups_test.nunique()),
                **metrics,
                **{f"best_{k}": v for k, v in best_params.items()},
            }
        )

        test_slice = model_df.iloc[test_idx][[EXPECTED_COLUMNS["id"], EXPECTED_COLUMNS["group"], EXPECTED_COLUMNS["team_member"]]].copy()
        test_slice["target"] = target
        test_slice["y_true"] = y_test.to_numpy()
        test_slice["y_pred"] = preds
        test_slice["residual"] = test_slice["y_true"] - test_slice["y_pred"]
        test_slice["outer_fold"] = outer_fold
        pred_rows.append(test_slice)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    trials_df = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame()

    global_best_params, global_trials_df = run_inner_optuna(
        X_train=X,
        y_train=y,
        groups_train=groups,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_trials=n_trials,
        n_splits=min(inner_splits, int(groups.nunique())),
    )
    if not global_trials_df.empty:
        global_trials_df.insert(0, "target", target)
        global_trials_df.insert(1, "outer_fold", "GLOBAL_FULL_DATA")
        trials_df = pd.concat([trials_df, global_trials_df], ignore_index=True)

    global_summary = {
        "target": target,
        "best_params": global_best_params,
        "mean_outer_r2": float(metrics_df["r2"].mean()) if not metrics_df.empty else np.nan,
        "mean_outer_mae": float(metrics_df["mae"].mean()) if not metrics_df.empty else np.nan,
        "mean_outer_rmse": float(metrics_df["rmse"].mean()) if not metrics_df.empty else np.nan,
        "mean_outer_spearman": float(metrics_df["spearman_rho"].mean()) if not metrics_df.empty else np.nan,
    }
    return metrics_df, predictions_df, trials_df, global_summary


def final_model_permutation_importance(bundle: ModelBundle) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # permutation importance sobre features crudas del pipeline completo
    from sklearn.inspection import permutation_importance

    perm_result = permutation_importance(
        bundle.pipeline,
        bundle.X_train_full,
        bundle.y_train_full,
        n_repeats=20,
        random_state=SEED,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    raw_perm_df = pd.DataFrame(
        {
            "target": bundle.target,
            "original_feature": bundle.feature_columns,
            "permutation_importance_mean": perm_result.importances_mean,
            "permutation_importance_std": perm_result.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)

    # importancia del modelo final sobre features transformadas
    model = bundle.pipeline.named_steps["model"]
    transformed_df = pd.DataFrame(
        {
            "target": bundle.target,
            "transformed_feature": bundle.transformed_feature_names,
            "original_feature": [bundle.transformed_to_original.get(name, name) for name in bundle.transformed_feature_names],
            "model_feature_importance": model.feature_importances_,
        }
    ).sort_values("model_feature_importance", ascending=False).reset_index(drop=True)

    return transformed_df, raw_perm_df


def write_master_excel(
    output_path: Path,
    bundle: DataBundle,
    corr_abs_df: pd.DataFrame,
    corr_rel_df: pd.DataFrame,
    corr_long_df: pd.DataFrame,
    formula_best_df: pd.DataFrame,
    formula_trials_df: pd.DataFrame,
    formula_cv_df: pd.DataFrame,
    model_metrics_df: pd.DataFrame,
    model_predictions_df: pd.DataFrame,
    model_trials_df: pd.DataFrame,
    model_best_params_df: pd.DataFrame,
    final_perm_transformed_df: pd.DataFrame,
    final_perm_agg_df: pd.DataFrame,
    plot_ready_formula_df: pd.DataFrame,
    plot_ready_metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        build_readme_sheet(bundle, output_dir).to_excel(writer, sheet_name="00_README", index=False)
        bundle.raw_df.to_excel(writer, sheet_name="01_INPUT_RAW", index=False)
        bundle.typed_df.to_excel(writer, sheet_name="02_INPUT_TYPED", index=False)
        bundle.derived_df.to_excel(writer, sheet_name="03_INPUT_DERIVED", index=False)
        bundle.column_audit_df.to_excel(writer, sheet_name="04_AUDIT_COLUMNS", index=False)
        bundle.range_audit_df.to_excel(writer, sheet_name="05_AUDIT_RANGES", index=False)
        corr_abs_df.to_excel(writer, sheet_name="06_CORR_ABSOLUTE", index=False)
        corr_rel_df.to_excel(writer, sheet_name="07_CORR_RELATIVE", index=False)
        formula_best_df.to_excel(writer, sheet_name="08_FORMULA_BEST", index=False)
        formula_trials_df.to_excel(writer, sheet_name="09_FORMULA_TRIALS", index=False)
        formula_cv_df.to_excel(writer, sheet_name="10_FORMULA_CV", index=False)
        model_best_params_df.to_excel(writer, sheet_name="11_MODEL_BEST_PARAMS", index=False)
        model_metrics_df.to_excel(writer, sheet_name="12_MODEL_OUTER_CV", index=False)
        model_predictions_df.to_excel(writer, sheet_name="13_MODEL_PREDS", index=False)
        model_trials_df.to_excel(writer, sheet_name="14_MODEL_TRIALS", index=False)
        final_perm_transformed_df.to_excel(writer, sheet_name="15_FINAL_PERM_TRANS", index=False)
        final_perm_agg_df.to_excel(writer, sheet_name="16_FINAL_PERM_AGG", index=False)
        corr_long_df.to_excel(writer, sheet_name="17_PLOT_CORR_LONG", index=False)
        plot_ready_formula_df.to_excel(writer, sheet_name="18_PLOT_FORMULA", index=False)
        plot_ready_metrics_df.to_excel(writer, sheet_name="19_PLOT_METRICS", index=False)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Path]:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    output_dirs = ensure_output_dirs(Path(args.output_dir))
    bundle = load_data_bundle(input_path)

    corr_abs_df, corr_rel_df, corr_long_df = get_correlation_tables(bundle.derived_df)

    formula_best_rows: List[Dict[str, Any]] = []
    formula_trials_frames: List[pd.DataFrame] = []
    formula_cv_frames: List[pd.DataFrame] = []
    plot_formula_frames: List[pd.DataFrame] = []
    for target in [t for t in MAIN_TARGETS if t in bundle.derived_df.columns]:
        best_summary, trials_df, fold_df = optimize_formula_weights(
            df=bundle.derived_df,
            target=target,
            n_trials=args.formula_trials,
            n_splits=min(args.outer_splits, int(bundle.derived_df[EXPECTED_COLUMNS["group"]].nunique())),
        )
        formula_best_rows.append(best_summary)
        if not trials_df.empty:
            formula_trials_frames.append(trials_df)
        if not fold_df.empty:
            formula_cv_frames.append(fold_df)
        plot_formula_frames.append(pd.DataFrame([best_summary]))

    formula_best_df = pd.DataFrame(formula_best_rows).sort_values("best_value", ascending=False).reset_index(drop=True)
    formula_trials_df = pd.concat(formula_trials_frames, ignore_index=True) if formula_trials_frames else pd.DataFrame()
    formula_cv_df = pd.concat(formula_cv_frames, ignore_index=True) if formula_cv_frames else pd.DataFrame()
    plot_formula_df = pd.concat(plot_formula_frames, ignore_index=True) if plot_formula_frames else pd.DataFrame()

    model_metrics_frames: List[pd.DataFrame] = []
    model_predictions_frames: List[pd.DataFrame] = []
    model_trials_frames: List[pd.DataFrame] = []
    best_params_rows: List[Dict[str, Any]] = []
    final_perm_transformed_frames: List[pd.DataFrame] = []
    final_perm_agg_frames: List[pd.DataFrame] = []

    for target in [t for t in MAIN_TARGETS if t in bundle.derived_df.columns]:
        metrics_df, preds_df, trials_df, global_summary = evaluate_model_nested_cv(
            derived_df=bundle.derived_df,
            target=target,
            n_trials=args.model_trials,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
        )
        if not metrics_df.empty:
            model_metrics_frames.append(metrics_df)
        if not preds_df.empty:
            model_predictions_frames.append(preds_df)
        if not trials_df.empty:
            model_trials_frames.append(trials_df)

        best_params_row = {"target": target, **global_summary["best_params"]}
        best_params_row.update(
            {
                "mean_outer_r2": global_summary["mean_outer_r2"],
                "mean_outer_mae": global_summary["mean_outer_mae"],
                "mean_outer_rmse": global_summary["mean_outer_rmse"],
                "mean_outer_spearman": global_summary["mean_outer_spearman"],
            }
        )
        best_params_rows.append(best_params_row)

        final_bundle = fit_final_model_bundle(bundle.derived_df, target, global_summary["best_params"])
        model_path = output_dirs["models"] / f"model_{target}.joblib"
        save_model_bundle(final_bundle, model_path)

        perm_transformed_df, perm_agg_df = final_model_permutation_importance(final_bundle)
        final_perm_transformed_frames.append(perm_transformed_df)
        final_perm_agg_frames.append(perm_agg_df)

    model_metrics_df = pd.concat(model_metrics_frames, ignore_index=True) if model_metrics_frames else pd.DataFrame()
    model_predictions_df = pd.concat(model_predictions_frames, ignore_index=True) if model_predictions_frames else pd.DataFrame()
    model_trials_df = pd.concat(model_trials_frames, ignore_index=True) if model_trials_frames else pd.DataFrame()
    model_best_params_df = pd.DataFrame(best_params_rows).sort_values("mean_outer_mae", ascending=True).reset_index(drop=True)
    final_perm_transformed_df = pd.concat(final_perm_transformed_frames, ignore_index=True) if final_perm_transformed_frames else pd.DataFrame()
    final_perm_agg_df = pd.concat(final_perm_agg_frames, ignore_index=True) if final_perm_agg_frames else pd.DataFrame()

    plot_ready_metrics_df = (
        model_metrics_df.groupby("target", as_index=False)[["r2", "mae", "rmse", "spearman_rho"]]
        .agg(["mean", "std"])
        .reset_index()
    ) if not model_metrics_df.empty else pd.DataFrame()
    if not plot_ready_metrics_df.empty:
        plot_ready_metrics_df.columns = [
            "target" if c == ("target", "") else f"{c[0]}_{c[1]}" for c in plot_ready_metrics_df.columns
        ]

    master_xlsx = output_dirs["reports"] / DEFAULT_MASTER_XLSX
    write_master_excel(
        output_path=master_xlsx,
        bundle=bundle,
        corr_abs_df=corr_abs_df,
        corr_rel_df=corr_rel_df,
        corr_long_df=corr_long_df,
        formula_best_df=formula_best_df,
        formula_trials_df=formula_trials_df,
        formula_cv_df=formula_cv_df,
        model_metrics_df=model_metrics_df,
        model_predictions_df=model_predictions_df,
        model_trials_df=model_trials_df,
        model_best_params_df=model_best_params_df,
        final_perm_transformed_df=final_perm_transformed_df,
        final_perm_agg_df=final_perm_agg_df,
        plot_ready_formula_df=plot_formula_df,
        plot_ready_metrics_df=plot_ready_metrics_df,
        output_dir=output_dirs["base"],
    )

    # CSV/JSON auxiliares
    formula_best_df.to_csv(output_dirs["tables"] / "formula_best.csv", index=False)
    model_best_params_df.to_csv(output_dirs["tables"] / "model_best_params.csv", index=False)
    model_metrics_df.to_csv(output_dirs["tables"] / "model_outer_cv_metrics.csv", index=False)
    model_predictions_df.to_csv(output_dirs["tables"] / "model_outer_predictions.csv", index=False)
    corr_long_df.to_csv(output_dirs["tables"] / "correlations_long.csv", index=False)

    json_dump(bundle.metadata, output_dirs["logs"] / "dataset_metadata.json")
    json_dump(
        {
            "input": str(input_path),
            "master_excel": str(master_xlsx),
            "model_trials": args.model_trials,
            "formula_trials": args.formula_trials,
            "outer_splits": args.outer_splits,
            "inner_splits": args.inner_splits,
        },
        output_dirs["logs"] / "run_config.json",
    )

    return {
        "output_base": output_dirs["base"],
        "master_excel": master_xlsx,
        "tables_dir": output_dirs["tables"],
        "models_dir": output_dirs["models"],
    }


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args)
    print("Pipeline finalizado.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
