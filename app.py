# -*- coding: utf-8 -*-
# Streamlit app for stacking calculator + SHAP force plot (level-2) - working dir version

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st

# ----------------------------
# numpy old alias shims (for SHAP / older libs)
# ----------------------------
# Some older SHAP / 3rd-party code uses deprecated numpy aliases.
for _name, _py in [("int", int), ("float", float), ("bool", bool)]:
    if not hasattr(np, _name):
        setattr(np, _name, _py)

# =========================
# Paths (relative to this file)
# =========================
BASE   = Path(__file__).parent.resolve()
OUTDIR = BASE / "stacking_outputs_full"
PERDIR = OUTDIR / "per_model"
DATA_PATH = BASE / "data.csv"

st.set_page_config(page_title="Stacking Level-2 Risk Calculator (with SHAP)", layout="wide")
st.title("📈 Stacking Level-2 Risk Calculator (with SHAP)")

TARGET_COL = "Result"
CAT_COLS = ["Surgical_Type", "Surgical_Segment", "Surgical_Approach", "VP_Use"]

# =========================
# EXACT feature set (12) + display names + units
# =========================
FEATURE_ORDER = [
    "Apacheii", "SOFA", "Temp",
    "Surgical_Type", "Surgical_Approach",
    "BUN", "Surgical_Segment",
    "SBP", "Platelets", "INR", "CCI", "VP_Use",
]

DISPLAY_NAME = {
    "Apacheii": "APACHE-II (score)",
    "SOFA": "SOFA (score)",
    "Temp": "Temp (°C)",
    "Surgical_Type": "Surgical Type",
    "Surgical_Approach": "Surgical Approach",
    "BUN": "BUN (mg/dL)",
    "Surgical_Segment": "Surgical Segment",
    "SBP": "SBP (mmHg)",
    "Platelets": "Platelets (10^9/L)",
    "INR": "INR (ratio)",
    "CCI": "CCI (score)",
    "VP_Use": "VP-Use",
}
def disp(col: str) -> str:
    return DISPLAY_NAME.get(col, col)

# =========================
# Categorical encodings (UI labels -> numeric codes)
# =========================
CAT_LABELS = {
    "Surgical_Type": {1: "Fusion", 2: "Non-Fusion"},
    "Surgical_Approach": {1: "Anterior", 2: "Posterior", 3: "Anterior + Posterior"},
    "Surgical_Segment": {1: "<4 Segments", 2: "4-8 Segments", 3: ">8 Segments"},
    "VP_Use": {0: "None-Use", 1: "Use"},
}
CAT_LABEL2CODE = {col: {label: code for code, label in m.items()} for col, m in CAT_LABELS.items()}

# =========================
# XGBoost compatibility patch
# =========================
def _ensure_xgb_compat(obj):
    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = tuple()
        XGBoostPresent = False
    else:
        XGBoostPresent = True

    def _patch_xgb(est):
        defaults = {
            "gpu_id": None,
            "predictor": None,
            "n_estimators": getattr(est, "n_estimators", None),
            "use_label_encoder": False,
            "enable_categorical": False,
        }
        for k, v in defaults.items():
            if not hasattr(est, k):
                try:
                    setattr(est, k, v)
                except Exception:
                    pass
        return est

    if not XGBoostPresent:
        return obj

    if hasattr(obj, "named_steps"):
        if "clf" in obj.named_steps and isinstance(obj.named_steps["clf"], XGBClassifier):
            _patch_xgb(obj.named_steps["clf"])
        for _, step in obj.named_steps.items():
            if isinstance(step, XGBClassifier):
                _patch_xgb(step)
    elif isinstance(obj, XGBClassifier) or "xgboost" in str(type(obj)).lower():
        _patch_xgb(obj)
    return obj

def _predict_pipe(pipe, X):
    """Predict prob with xgboost/catboost compat, column alignment, dtype fixes."""
    import pandas as pd
    pipe = _ensure_xgb_compat(pipe)
    est = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") and "clf" in pipe.named_steps else pipe
    est_name = type(est).__name__.lower()

    X_in = X.copy()

    # CatBoost: align by feature_names_ and cast categoricals to string
    try:
        import catboost  # noqa: F401
        is_cat = ("catboost" in est.__module__.lower()) or ("catboost" in est_name)
    except Exception:
        is_cat = False

    if is_cat:
        feat_names = getattr(est, "feature_names_", None)
        if isinstance(feat_names, (list, tuple)) and len(feat_names) > 0:
            for c in feat_names:
                if c not in X_in.columns:
                    X_in[c] = np.nan
            X_in = X_in.loc[:, list(feat_names)]
        for c in [c for c in CAT_COLS if c in X_in.columns]:
            if pd.api.types.is_float_dtype(X_in[c]):
                X_in[c] = X_in[c].astype("Int64").astype(str)
            elif pd.api.types.is_integer_dtype(X_in[c]):
                X_in[c] = X_in[c].astype(str)
            else:
                X_in[c] = X_in[c].astype(str)
        proba = est.predict_proba(X_in)
    else:
        try:
            proba = pipe.predict_proba(X_in)
        except Exception:
            proba = est.predict_proba(X_in)

    proba = np.asarray(proba)
    if proba.ndim == 1:
        return float(proba.ravel()[0])
    if proba.shape[1] >= 2:
        return float(proba[0, 1])
    return float(np.ravel(proba)[-1])

# =========================
# Loading data and models
# =========================
@st.cache_data
def load_data():
    # Use data.csv just to derive UI ranges / categories
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    # 1) Level-2 logistic meta model
    meta = joblib.load(OUTDIR / "meta_lr_cv.pkl") if (OUTDIR / "meta_lr_cv.pkl").exists() else joblib.load(BASE / "meta_lr_cv.pkl")

    # 2) Level-1 OOF predictions and preferred column order
    oof = pd.read_csv(OUTDIR / "level1_oof_predictions.csv")
    all_oof_cols = [c for c in oof.columns if c != "y_train"]

    meta_feature_order = None
    meta_feat_file = OUTDIR / "meta_features.json"
    if meta_feat_file.exists():
        try:
            with open(meta_feat_file, "r", encoding="utf-8") as f:
                js = json.load(f)
            cols_from_file = js.get("meta_feature_columns", [])
            meta_feature_order = [c for c in cols_from_file if c in all_oof_cols]
        except Exception:
            meta_feature_order = None
    if not meta_feature_order:
        meta_feature_order = all_oof_cols

    # Align with meta.feature_names_in_ if present
    feat_in = getattr(meta, "feature_names_in_", None)
    if feat_in is not None:
        feat_in = list(feat_in)
        meta_feature_order = [c for c in meta_feature_order if c in feat_in]

    # Column means for imputing missing Level-1 models
    col_means = oof[meta_feature_order].mean().to_dict()

    # Load available Level-1 models saved as joblib
    avail = {}
    for p in sorted(PERDIR.glob("model_*.pkl")):
        name = p.stem.replace("model_", "")
        if name in meta_feature_order:
            try:
                mdl = joblib.load(p)
                avail[name] = _ensure_xgb_compat(mdl)
            except Exception as e:
                print(f"[WARN] loading {p.name} failed: {repr(e)}")

    return meta, avail, meta_feature_order, col_means

df = load_data()
# Only keep the features this app needs
raw_cols = [c for c in FEATURE_ORDER if c in df.columns]
num_cols = [c for c in raw_cols if c not in CAT_COLS]

meta, base_models, meta_feature_order, col_means = load_models()

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("⚙️ Inputs")
with st.sidebar:
    # Robust numeric ranges from data.csv
    qlo = df[num_cols].quantile(0.01) if len(num_cols) else pd.Series(dtype=float)
    qhi = df[num_cols].quantile(0.99) if len(num_cols) else pd.Series(dtype=float)

    inputs = {}
    for col in raw_cols:
        if col in CAT_COLS:
            if col in CAT_LABELS and len(CAT_LABELS[col]) > 0:
                labels = list(CAT_LABELS[col].values())
                selected_label = st.selectbox(disp(col), labels, index=0, key=f"sel_{col}")
                code = CAT_LABEL2CODE[col].get(selected_label, list(CAT_LABELS[col].keys())[0])
                inputs[col] = int(code)
            else:
                # fallback to category values found in data.csv
                cats = sorted(map(str, pd.Series(df[col].astype("object")).dropna().unique().tolist()))
                default = cats[0] if cats else ""
                idx = cats.index(default) if default in cats else 0
                inputs[col] = st.selectbox(disp(col), cats, index=idx, key=f"sel_{col}")
        else:
            series = df[col]
            val = float(series.median()) if pd.api.types.is_numeric_dtype(series) else 0.0
            lo  = float(qlo.get(col, series.min()))
            hi  = float(qhi.get(col, series.max()))
            step = (hi - lo) / 100 if (hi > lo) else 1.0
            inputs[col] = st.number_input(disp(col), value=float(val), min_value=float(lo), max_value=float(hi),
                                          step=float(step), format="%.3f", key=f"num_{col}")

    go = st.button("🔮 Predict & Explain", type="primary")

# =========================
# Prediction
# =========================
def build_input_df(inputs_dict):
    row = {c: inputs_dict[c] for c in raw_cols}
    X1 = pd.DataFrame([row], columns=raw_cols)
    for c in CAT_COLS:
        if c in X1.columns:
            if isinstance(X1.at[0, c], (int, np.integer)):
                X1[c] = X1[c].astype(int)
            else:
                X1[c] = X1[c].astype("object")
    return X1

def predict_level1_meta(X1: pd.DataFrame):
    feats, filled_flag = {}, {}
    for name in meta_feature_order:
        if name in base_models:
            try:
                feats[name] = _predict_pipe(base_models[name], X1)
                filled_flag[name] = False
            except Exception:
                feats[name] = float(col_means.get(name, 0.5))
                filled_flag[name] = True
        else:
            feats[name] = float(col_means.get(name, 0.5))
            filled_flag[name] = True

    X_meta_one = pd.DataFrame([[feats[c] for c in meta_feature_order]],
                              columns=meta_feature_order)

    feat_in = getattr(meta, "feature_names_in_", None)
    if feat_in is not None:
        # Ensure the exact order expected by the meta model
        X_meta_one = X_meta_one.loc[:, list(feat_in)]

    p_hat = float(meta.predict_proba(X_meta_one)[0, 1])
    try:
        logit = float(meta.decision_function(X_meta_one)[0])
    except Exception:
        p = np.clip(p_hat, 1e-12, 1-1e-12)
        logit = float(np.log(p/(1-p)))

    return p_hat, logit, X_meta_one

# =========================
# SHAP explainer (Level-2)
# =========================
@st.cache_resource
def build_level2_explainer():
    oof = pd.read_csv(OUTDIR / "level1_oof_predictions.csv")
    bg = oof.drop(columns=["y_train"])

    # Align with meta feature order if available
    feat_in = getattr(meta, "feature_names_in_", None)
    if feat_in is not None:
        cols = [c for c in feat_in if c in bg.columns]
        if cols:
            bg = bg[cols]

    if len(bg) > 500:
        bg = bg.sample(500, random_state=2025)

    # Prefer LinearExplainer; fallback to KernelExplainer
    try:
        expl = shap.LinearExplainer(meta, bg, feature_perturbation="interventional")
    except Exception:
        expl = shap.KernelExplainer(
            lambda X: meta.predict_proba(pd.DataFrame(X, columns=bg.columns))[:, 1],
            bg
        )
    return expl

# =========================
# Vertical layout: Sepsis probability + SHAP Force Plot (matplotlib, compact)
# =========================
if go:
    X1 = build_input_df(inputs)
    p_hat, logit, X_meta_one = predict_level1_meta(X1)

    st.subheader("🎯 Predicted Risk")
    st.metric("Sepsis probability (Level-2)", f"{p_hat:.1%}")
    st.caption(f"Logit = {logit:.3f}")

    st.divider()

    st.subheader("🧠 SHAP Force Plot (Level-2: contributions of Level-1 models)")
    expl = build_level2_explainer()

    values = None
    base = None
    try:
        sv = expl(X_meta_one)  # shap>=0.40: Explanation
        base = float(np.ravel(sv.base_values)[0])
        values = np.ravel(sv.values)
    except Exception:
        try:
            sv = expl.shap_values(X_meta_one)  # legacy API
            if isinstance(sv, list):
                values = np.array(sv[1][0])  # positive class
            else:
                values = np.array(sv[0])
            base = float(getattr(meta, "intercept_", [[0]])[0][0])
        except Exception as e:
            st.error(f"SHAP computation failed: {repr(e)}")

    if values is not None and base is not None:
        # Round feature inputs to 2 decimals for the force plot
        features_2dec = np.round(X_meta_one.iloc[0].values.astype(float), 2)

        import matplotlib.pyplot as plt
        plt.clf()
        plt.figure(figsize=(10, 2.2))
        shap.force_plot(
            base_value=base,
            shap_values=values,
            features=features_2dec,
            feature_names=X_meta_one.columns.tolist(),
            matplotlib=True,
            show=False
        )
        fig = plt.gcf()
        plt.margins(x=0.02)
        plt.tight_layout()
        st.pyplot(fig, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        st.info("SHAP force plot unavailable.")

    st.success("Done. Update inputs on the left to recompute.")

else:
    st.info("Fill in inputs on the left, then click **Predict & Explain**.")
    st.caption(
        "The force plot explains the Level-2 logistic model using Level-1 model outputs as features. "
    )
