from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
DEFAULT_INPUT = "consolidado_ordenado.xlsx"
DEFAULT_OUTPUT_DIR = "prestige_outputs"
DEFAULT_MASTER_XLSX = "prestige_master.xlsx"

EXPECTED_COLUMNS: Dict[str, str] = {
    "id": "ID",
    "group": "Group",
    "role": "Rol",
    "team_member": "Team Member",
    "gender": "Gender",
    "first_a": "First Speaker Activity A (0 No - 1 Si)",
    "speak_a": "Speaking Time Activity A",
    "first_b": "First Speaker Activity B (0 No - 1 Si)",
    "speak_b": "Speaking Time Activity B",
    "acad": "Normaliced Nota Promedio (0-1)",
    "exp_proj": "Normaliced en proyectos de software reales (0-1)",
    "exp_plan": "Normaliced Experiencia con Planning Poker o Planning Game (0-1)",
    "exp_model": "Normaliced Experiencia en modelado conceptual de software (0-1)",
    "peer": "Normaliced Promedio total Desempeño percibido Sin Autoevaluación (0-1)",
    "course_isf": "Cursó ISF (0 No - 1 Si)",
    "course_fa": "Cursó FA (0 No - 1 Si)",
}

PRESTIGE_SOURCE_KEYS = ["acad", "exp_proj", "exp_plan", "exp_model", "peer"]
PRESTIGE_SOURCE_COLUMNS = [EXPECTED_COLUMNS[k] for k in PRESTIGE_SOURCE_KEYS]

RE_TOTAL_ATTN = re.compile(
    r"^Total Visual Attention Received from the member Activity ([A-Z]) TM(\d+)$",
    flags=re.IGNORECASE,
)
RE_SPEAK_ATTN = re.compile(
    r"^Visual attention when speaking received from the member Activity ([A-Z]) TM(\d+)$",
    flags=re.IGNORECASE,
)
RE_SILENT_ATTN = re.compile(
    r"^Visual attention when not speaking received from the member Activity ([A-Z]) TM(\d+)$",
    flags=re.IGNORECASE,
)

MAIN_TARGETS = [
    "Gaze_While_Speaking",
    "Gaze_While_Silent",
    "Total_Speaking_Time",
    "Total_Attention_Received",
    "SNA_Eigenvector",
]

TARGET_LABELS = {
    "Gaze_While_Speaking": "Atención visual recibida mientras habla",
    "Gaze_While_Silent": "Atención visual recibida en silencio",
    "Total_Speaking_Time": "Tiempo total de habla",
    "Total_Attention_Received": "Atención visual total recibida",
    "SNA_Eigenvector": "Centralidad Eigenvector en red de miradas",
}


@dataclass
class DataBundle:
    raw_df: pd.DataFrame
    typed_df: pd.DataFrame
    derived_df: pd.DataFrame
    column_audit_df: pd.DataFrame
    range_audit_df: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class ModelBundle:
    target: str
    pipeline: Pipeline
    feature_columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    transformed_feature_names: List[str]
    transformed_to_original: Dict[str, str]
    best_params: Dict[str, Any]
    X_train_full: pd.DataFrame
    y_train_full: pd.Series
    groups_train_full: pd.Series


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "base": base_dir,
        "reports": base_dir / "reports",
        "models": base_dir / "models",
        "plots": base_dir / "plots",
        "shap": base_dir / "shap",
        "logs": base_dir / "logs",
        "tables": base_dir / "tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def normalize_colname(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip())


def find_matching_columns(columns: Iterable[str], pattern: re.Pattern[str]) -> List[str]:
    return sorted([col for col in columns if pattern.match(col)])


def conservative_type_cast(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    typed = df.copy()
    audit_rows: List[Dict[str, Any]] = []

    protected_text_cols = {EXPECTED_COLUMNS["role"], EXPECTED_COLUMNS["gender"]}

    for col in typed.columns:
        original = typed[col]
        original_dtype = str(original.dtype)
        inferred_type = "text"
        parse_ratio = np.nan
        n_non_null = int(original.notna().sum())

        if col in protected_text_cols:
            typed[col] = (
                original.astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .replace({"": pd.NA})
            )
            inferred_type = "protected_categorical"
        elif pd.api.types.is_numeric_dtype(original):
            typed[col] = pd.to_numeric(original, errors="coerce")
            inferred_type = "numeric_existing"
            parse_ratio = 1.0 if n_non_null else np.nan
        else:
            cleaned_str = (
                original.astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})
            )
            parsed = pd.to_numeric(
                cleaned_str.str.replace(",", ".", regex=False),
                errors="coerce",
            )
            non_empty = int(cleaned_str.notna().sum())
            parsed_count = int(parsed.notna().sum())
            parse_ratio = parsed_count / non_empty if non_empty else np.nan
            if non_empty > 0 and parse_ratio >= 0.85:
                typed[col] = parsed
                inferred_type = "numeric_inferred"
            else:
                typed[col] = cleaned_str
                inferred_type = "text"

        audit_rows.append(
            {
                "column": col,
                "original_dtype": original_dtype,
                "typed_dtype": str(typed[col].dtype),
                "inferred_type": inferred_type,
                "non_null_rows": n_non_null,
                "parse_ratio": parse_ratio,
                "n_unique": int(typed[col].nunique(dropna=True)),
            }
        )

    if EXPECTED_COLUMNS["role"] in typed.columns:
        typed["Rol_Clean"] = (
            typed[EXPECTED_COLUMNS["role"]]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    if EXPECTED_COLUMNS["gender"] in typed.columns:
        typed["Gender_Clean"] = (
            typed[EXPECTED_COLUMNS["gender"]]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    return typed, pd.DataFrame(audit_rows)


def validate_expected_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in EXPECTED_COLUMNS.values() if col not in df.columns]


def minmax_01(series: pd.Series, fit_min: Optional[float] = None, fit_max: Optional[float] = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=numeric.index, dtype="float64")
    min_val = float(valid.min()) if fit_min is None else float(fit_min)
    max_val = float(valid.max()) if fit_max is None else float(fit_max)
    if math.isclose(max_val, min_val):
        return pd.Series(0.0, index=numeric.index, dtype="float64")
    return (numeric - min_val) / (max_val - min_val)


def minmax_fit(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return {"min": 0.0, "max": 0.0}
    return {"min": float(valid.min()), "max": float(valid.max())}

def is_already_01(series: pd.Series, tol: float = 1e-9) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return False
    return bool(((valid >= -tol) & (valid <= 1.0 + tol)).all())


def ensure_01(series: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    numeric = pd.to_numeric(series, errors="coerce")
    if is_already_01(numeric):
        return numeric.astype("float64"), {
            "action": "kept_as_is",
            "min": float(numeric.min()) if numeric.notna().any() else np.nan,
            "max": float(numeric.max()) if numeric.notna().any() else np.nan,
        }
    fit = minmax_fit(numeric)
    return minmax_01(numeric, fit_min=fit["min"], fit_max=fit["max"]), {
        "action": "minmax_rescaled",
        "min": fit["min"],
        "max": fit["max"],
    }


def zscore_within_group(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=numeric.index, dtype="float64")
    std = float(valid.std(ddof=0))
    if math.isclose(std, 0.0):
        return pd.Series(0.0, index=numeric.index, dtype="float64")
    return (numeric - float(valid.mean())) / std


def safe_spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3:
        return np.nan, np.nan, len(pair)
    rho, p_value = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return float(rho), float(p_value), int(len(pair))


def sum_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if not columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return df[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


def build_attention_graph_metrics(df: pd.DataFrame, total_attention_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_col = EXPECTED_COLUMNS["group"]
    member_col = EXPECTED_COLUMNS["team_member"]
    id_col = EXPECTED_COLUMNS["id"]

    for group_id, group_df in df.groupby(group_col, dropna=True):
        group_df = group_df.copy()
        graph = nx.DiGraph()
        graph.add_nodes_from(group_df[member_col].tolist())

        for _, row in group_df.iterrows():
            target = row[member_col]
            for col in total_attention_cols:
                match = RE_TOTAL_ATTN.match(col)
                if not match:
                    continue
                source = int(match.group(2))
                weight = pd.to_numeric(row[col], errors="coerce")
                weight = 0.0 if pd.isna(weight) else float(weight)
                if source == target or weight <= 0:
                    continue
                if graph.has_edge(source, target):
                    graph[source][target]["weight"] += weight
                else:
                    graph.add_edge(source, target, weight=weight)

        try:
            eigen = nx.eigenvector_centrality(graph, weight="weight", max_iter=5000)
        except Exception:
            eigen = {node: np.nan for node in graph.nodes}
        try:
            pagerank = nx.pagerank(graph, weight="weight")
        except Exception:
            pagerank = {node: np.nan for node in graph.nodes}

        indegree = dict(graph.in_degree(weight="weight"))
        outdegree = dict(graph.out_degree(weight="weight"))

        for _, row in group_df.iterrows():
            tm = row[member_col]
            rows.append(
                {
                    id_col: row[id_col],
                    group_col: group_id,
                    member_col: tm,
                    "SNA_Eigenvector": eigen.get(tm, np.nan),
                    "SNA_PageRank": pagerank.get(tm, np.nan),
                    "SNA_InDegree_Weighted": indegree.get(tm, 0.0),
                    "SNA_OutDegree_Weighted": outdegree.get(tm, 0.0),
                    "SNA_Attention_Balance": indegree.get(tm, 0.0) - outdegree.get(tm, 0.0),
                }
            )

    return pd.DataFrame(rows)


def compute_derived_features(typed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = typed_df.copy()

    total_attention_cols = find_matching_columns(df.columns, RE_TOTAL_ATTN)
    speaking_attention_cols = find_matching_columns(df.columns, RE_SPEAK_ATTN)
    silent_attention_cols = find_matching_columns(df.columns, RE_SILENT_ATTN)
    speaking_time_cols = [c for c in [EXPECTED_COLUMNS["speak_a"], EXPECTED_COLUMNS["speak_b"]] if c in df.columns]

    if EXPECTED_COLUMNS["acad"] in df.columns:
        df["Prestige_Academic_Raw"] = pd.to_numeric(df[EXPECTED_COLUMNS["acad"]], errors="coerce")
    exp_cols = [EXPECTED_COLUMNS[k] for k in ["exp_proj", "exp_plan", "exp_model"] if EXPECTED_COLUMNS[k] in df.columns]
    if exp_cols:
        df["Prestige_Experiential_Raw"] = df[exp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if EXPECTED_COLUMNS["peer"] in df.columns:
        df["Prestige_Peer_Raw"] = pd.to_numeric(df[EXPECTED_COLUMNS["peer"]], errors="coerce")

    component_fits: Dict[str, Dict[str, Any]] = {}

    for source_col, raw_col, scaled_col in [
        (EXPECTED_COLUMNS["acad"], "Prestige_Academic_Raw", "Prestige_Academic_01"),
        (EXPECTED_COLUMNS["peer"], "Prestige_Peer_Raw", "Prestige_Peer_01"),
    ]:
        if source_col in df.columns:
            df[scaled_col], component_fits[scaled_col] = ensure_01(df[source_col])
    
    if exp_cols:
        exp_scaled_cols: List[str] = []
        for source_col in exp_cols:
            helper_col = f"{source_col}__01"
            df[helper_col], component_fits[helper_col] = ensure_01(df[source_col])
            exp_scaled_cols.append(helper_col)
        df["Prestige_Experiential_01"] = df[exp_scaled_cols].mean(axis=1)

    group_col = EXPECTED_COLUMNS["group"]
    for source, target in [
        ("Prestige_Academic_01", "Prestige_Academic_RelZ"),
        ("Prestige_Experiential_01", "Prestige_Experiential_RelZ"),
        ("Prestige_Peer_01", "Prestige_Peer_RelZ"),
    ]:
        if source in df.columns:
            df[target] = df.groupby(group_col, dropna=False)[source].transform(zscore_within_group)

    df["Total_Attention_Received"] = sum_columns(df, total_attention_cols)
    df["Gaze_While_Speaking"] = sum_columns(df, speaking_attention_cols)
    df["Gaze_While_Silent"] = sum_columns(df, silent_attention_cols)
    df["Total_Speaking_Time"] = sum_columns(df, speaking_time_cols)

    if EXPECTED_COLUMNS["first_a"] in df.columns and EXPECTED_COLUMNS["first_b"] in df.columns:
        df["First_Speaker_Count"] = (
            pd.to_numeric(df[EXPECTED_COLUMNS["first_a"]], errors="coerce").fillna(0.0)
            + pd.to_numeric(df[EXPECTED_COLUMNS["first_b"]], errors="coerce").fillna(0.0)
        )

    for source, target in [
        ("Total_Attention_Received", "Attention_Received_Share_Group"),
        ("Gaze_While_Speaking", "Gaze_Speaking_Share_Group"),
        ("Gaze_While_Silent", "Gaze_Silent_Share_Group"),
        ("Total_Speaking_Time", "Speaking_Time_Share_Group"),
    ]:
        totals = df.groupby(group_col, dropna=False)[source].transform("sum")
        df[target] = np.where(totals != 0, df[source] / totals, np.nan)

    graph_metrics = build_attention_graph_metrics(df, total_attention_cols)
    if not graph_metrics.empty:
        df = df.merge(
            graph_metrics,
            on=[EXPECTED_COLUMNS["id"], EXPECTED_COLUMNS["group"], EXPECTED_COLUMNS["team_member"]],
            how="left",
        )

    range_audit_rows: List[Dict[str, Any]] = []
    for source_col in PRESTIGE_SOURCE_COLUMNS:
        if source_col not in df.columns:
            continue
        numeric = pd.to_numeric(df[source_col], errors="coerce")
        range_audit_rows.append(
            {
                "column": source_col,
                "min_original": float(numeric.min()) if numeric.notna().any() else np.nan,
                "max_original": float(numeric.max()) if numeric.notna().any() else np.nan,
                "mean_original": float(numeric.mean()) if numeric.notna().any() else np.nan,
                "n_missing": int(numeric.isna().sum()),
                "rows_lt_0": int((numeric < 0).fillna(False).sum()),
                "rows_gt_1": int((numeric > 1).fillna(False).sum()),
                "looks_normalized_0_1": bool(((numeric >= 0) & (numeric <= 1) | numeric.isna()).all()),
            }
        )

    metadata = {
        "detected_total_attention_columns": total_attention_cols,
        "detected_speaking_attention_columns": speaking_attention_cols,
        "detected_silent_attention_columns": silent_attention_cols,
        "detected_speaking_time_columns": speaking_time_cols,
        "component_scalers": component_fits,
    }
    return df, pd.DataFrame(range_audit_rows), metadata


def load_data_bundle(input_path: Path) -> DataBundle:
    raw_df = pd.read_excel(input_path)
    raw_df.columns = [normalize_colname(c) for c in raw_df.columns]
    typed_df, column_audit_df = conservative_type_cast(raw_df)
    missing = validate_expected_columns(typed_df)
    if missing:
        raise ValueError(f"Faltan columnas esperadas en el input: {missing}")
    derived_df, range_audit_df, metadata = compute_derived_features(typed_df)
    metadata = {
        **metadata,
        "input_rows": int(len(raw_df)),
        "input_columns": int(raw_df.shape[1]),
        "groups": int(derived_df[EXPECTED_COLUMNS["group"]].nunique()),
        "group_sizes": derived_df.groupby(EXPECTED_COLUMNS["group"]).size().to_dict(),
        "missing_expected_columns": missing,
    }
    return DataBundle(
        raw_df=raw_df,
        typed_df=typed_df,
        derived_df=derived_df,
        column_audit_df=column_audit_df,
        range_audit_df=range_audit_df,
        metadata=metadata,
    )


def get_correlation_tables(derived_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    abs_features = [c for c in ["Prestige_Academic_01", "Prestige_Experiential_01", "Prestige_Peer_01"] if c in derived_df.columns]
    rel_features = [c for c in ["Prestige_Academic_RelZ", "Prestige_Experiential_RelZ", "Prestige_Peer_RelZ"] if c in derived_df.columns]
    targets = [c for c in MAIN_TARGETS + ["SNA_PageRank"] if c in derived_df.columns]

    rows: List[Dict[str, Any]] = []
    for family, features in [("absolute", abs_features), ("relative", rel_features)]:
        for feature in features:
            for target in targets:
                rho, p_value, n_obs = safe_spearman(derived_df[feature], derived_df[target])
                rows.append(
                    {
                        "family": family,
                        "feature": feature,
                        "target": target,
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "n": n_obs,
                        "abs_rho": abs(rho) if pd.notna(rho) else np.nan,
                    }
                )
    corr_long = pd.DataFrame(rows).sort_values(["family", "abs_rho"], ascending=[True, False]).reset_index(drop=True)
    corr_abs = corr_long[corr_long["family"] == "absolute"].copy()
    corr_rel = corr_long[corr_long["family"] == "relative"].copy()
    return corr_abs, corr_rel, corr_long


def get_feature_spec(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric = [
        "Prestige_Academic_01",
        "Prestige_Experiential_01",
        "Prestige_Peer_01",
        "Prestige_Academic_RelZ",
        "Prestige_Experiential_RelZ",
        "Prestige_Peer_RelZ",
        EXPECTED_COLUMNS["course_isf"],
        EXPECTED_COLUMNS["course_fa"],
        "First_Speaker_Count",
    ]
    categorical = ["Gender_Clean", "Rol_Clean"]
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    return numeric + categorical, numeric, categorical


def build_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                list(numeric_features),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model(params: Mapping[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 400)),
        max_depth=None if params.get("max_depth") in (None, "None") else int(params["max_depth"]),
        min_samples_split=int(params.get("min_samples_split", 2)),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        max_features=params.get("max_features", "sqrt"),
        bootstrap=bool(params.get("bootstrap", True)),
        criterion=params.get("criterion", "squared_error"),
        random_state=SEED,
        n_jobs=1,
    )


def build_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    model_params: Mapping[str, Any],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", build_model(model_params)),
        ]
    )


def group_kfold_splits(groups: pd.Series, n_splits: int = 5) -> GroupKFold:
    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise ValueError("Se requieren al menos 2 grupos para GroupKFold.")
    return GroupKFold(n_splits=min(n_splits, n_groups))


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rho, p_value, _ = safe_spearman(pd.Series(y_true), pd.Series(y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman_rho": rho,
        "spearman_p_value": p_value,
    }


def precompute_feature_name_mapping(
    preprocessor: ColumnTransformer,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Tuple[List[str], Dict[str, str]]:
    feature_names = list(preprocessor.get_feature_names_out())
    mapping: Dict[str, str] = {}

    for name in feature_names:
        if name.startswith("num__"):
            mapping[name] = name.replace("num__", "", 1)
        elif name.startswith("cat__"):
            without_prefix = name.replace("cat__", "", 1)
            matched = None
            for candidate in categorical_features:
                if without_prefix.startswith(candidate + "_") or without_prefix == candidate:
                    matched = candidate
                    break
            mapping[name] = matched or without_prefix
        else:
            mapping[name] = name
    return feature_names, mapping


def aggregate_permutation_importance(importances: np.ndarray, feature_names: Sequence[str], mapping: Mapping[str, str]) -> pd.DataFrame:
    rows = [
        {
            "transformed_feature": feature_name,
            "original_feature": mapping.get(feature_name, feature_name),
            "importance": float(importance),
        }
        for feature_name, importance in zip(feature_names, importances)
    ]
    df = pd.DataFrame(rows)
    return (
        df.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def json_dump(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_model_bundle(bundle: ModelBundle, path: Path) -> None:
    payload = {
        "target": bundle.target,
        "pipeline": bundle.pipeline,
        "feature_columns": bundle.feature_columns,
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "transformed_feature_names": bundle.transformed_feature_names,
        "transformed_to_original": bundle.transformed_to_original,
        "best_params": bundle.best_params,
        "X_train_full": bundle.X_train_full,
        "y_train_full": bundle.y_train_full,
        "groups_train_full": bundle.groups_train_full,
    }
    joblib.dump(payload, path)


def load_model_bundle(path: Path) -> ModelBundle:
    payload = joblib.load(path)
    return ModelBundle(
        target=payload["target"],
        pipeline=payload["pipeline"],
        feature_columns=payload["feature_columns"],
        numeric_features=payload["numeric_features"],
        categorical_features=payload["categorical_features"],
        transformed_feature_names=payload["transformed_feature_names"],
        transformed_to_original=payload["transformed_to_original"],
        best_params=payload["best_params"],
        X_train_full=payload["X_train_full"],
        y_train_full=payload["y_train_full"],
        groups_train_full=payload["groups_train_full"],
    )


def build_readme_sheet(bundle: DataBundle, output_dir: Path) -> pd.DataFrame:
    rows = [
        {"section": "input_file", "detail": DEFAULT_INPUT},
        {"section": "rows", "detail": bundle.metadata["input_rows"]},
        {"section": "columns", "detail": bundle.metadata["input_columns"]},
        {"section": "groups", "detail": bundle.metadata["groups"]},
        {"section": "output_dir", "detail": str(output_dir)},
        {
            "section": "preservation",
            "detail": "Se preserva el input crudo, una versión tipada y una versión derivada; no se eliminan columnas originales.",
        },
        {
            "section": "validation",
            "detail": "La validación predictiva usa GroupKFold para evitar mezclar miembros del mismo equipo entre entrenamiento y prueba.",
        },
        {
            "section": "bayesian_optimization",
            "detail": "Optuna se usa para pesos de fórmula de prestigio y para hiperparámetros de RandomForestRegressor.",
        },
        {
            "section": "shap",
            "detail": "Las explicaciones SHAP se calculan sobre modelos finales entrenados con el pipeline reproducible y se exportan en tablas y figuras.",
        },
    ]
    return pd.DataFrame(rows)
