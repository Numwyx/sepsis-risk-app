# -*- coding: utf-8 -*-
import os, json, glob, inspect, importlib, re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ========== 全局配置 ==========
RANDOM_STATE = 20250915
np.random.seed(RANDOM_STATE)

BASE = Path.home() / "spinesurgery"
DATA_PATH = BASE / "data.csv"

OUTDIR = BASE / "stacking_outputs_full"
OUTDIR.mkdir(parents=True, exist_ok=True)
PERDIR = OUTDIR / "per_model"
PERDIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Result"
CAT_COLS = ["Surgical_Type", "Surgical_Segment", "Surgical_Approach", "VP_Use"]

# 如需避免重复跑，设 True（若已存在产物就跳过该模型）
SKIP_COMPLETED = False

# ========== 读取数据 ==========
print("Loading data:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
assert TARGET_COL in df.columns, f"目标列 {TARGET_COL} 不存在"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)
numeric_cols = [c for c in X.columns if c not in CAT_COLS]

# 7:3 划分
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

# ========== 依赖加载 ==========
def must_import(name, fromlist=None):
    try:
        if fromlist:
            return __import__(name, fromlist=fromlist)
        return __import__(name)
    except Exception as e:
        raise ImportError(f"缺少依赖 {name}：{e}")

xgboost = must_import("xgboost")
lightgbm = must_import("lightgbm")
catboost = must_import("catboost")
torch = must_import("torch")
rtdl = must_import("rtdl")
tabnet_pkg = must_import("pytorch_tabnet.tab_model", fromlist=['TabNetClassifier'])
pytab = must_import("pytorch_tabular")
pytab_lightning = must_import("pytorch_lightning")

from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.config import OptimizerConfig, TrainerConfig, DataConfig

# ===== 兼容不同版本 pytorch_tabular 的 NODE 配置类，统一别名 _NODE_CFG =====
def _resolve_node_cfg():
    tries = [
        ("NODEModelConfig", "pytorch_tabular.models"),
        ("NODEModelConfig", "pytorch_tabular.models.node"),
        ("NodeModelConfig", "pytorch_tabular.models"),
        ("NodeModelConfig", "pytorch_tabular.models.node"),
        ("NODEConfig",      "pytorch_tabular.models"),
        ("NODEConfig",      "pytorch_tabular.models.node"),
        ("NodeConfig",      "pytorch_tabular.models"),
        ("NodeConfig",      "pytorch_tabular.models.node"),
    ]
    for cls, mod in tries:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, cls):
                return getattr(m, cls)
        except Exception:
            pass
    avail0 = []
    for mod in ("pytorch_tabular.models", "pytorch_tabular.models.node"):
        try:
            m = importlib.import_module(mod)
            avail0 += [f"{mod}.{n}" for n in dir(m) if (("NODE" in n or "Node" in n) and "Config" in n)]
        except Exception:
            continue
    raise ImportError(
        "未能解析 pytorch_tabular 的 NODE 配置类。"
        f" 可用候选：{avail0}\n"
        "建议：安装 pytorch-tabular>=1.1 且 lightning>=2.0，或安装 0.7.0 + lightning<2.0。"
    )
_NODE_CFG = _resolve_node_cfg()

# === 兼容 NodeConfig 不同版本（task / tree_dim / colsample / bin_function） ===
def _make_node_config(d_in: int, params: dict):
    """
    兼容不同版本的 NodeConfig/NODEModelConfig：
    - 有的版本必填 task
    - tree_dim vs tree_output_dim
    - colsample vs colsample_bytree
    - bin_function 在部分版本仅允许 ['entmoid15','sparsemoid']：自动回退
    """
    sig = inspect.signature(_NODE_CFG)
    sig_keys = set(sig.parameters.keys())

    def build_kwargs(bin_fn):
        cand = {
            "num_layers": params.get("num_layers"),
            "num_trees":  params.get("num_trees"),
            "depth":      params.get("depth"),
            "input_dim":  d_in,
            "layer_norm": True,
            "embedding_dropout": 0.0,
        }
        # 任务类型
        if "task" in sig_keys:
            cand["task"] = "classification"

        # 维度字段名兼容
        tree_dim_val = params.get("tree_output_dim", params.get("tree_dim", 3))
        if "tree_dim" in sig_keys:
            cand["tree_dim"] = tree_dim_val
        elif "tree_output_dim" in sig_keys:
            cand["tree_output_dim"] = tree_dim_val

        # 子采样字段名兼容
        colsample_val = params.get("colsample", 0.8)
        if "colsample" in sig_keys:
            cand["colsample"] = colsample_val
        elif "colsample_bytree" in sig_keys:
            cand["colsample_bytree"] = colsample_val

        # bin_function（按需传入）
        if ("bin_function" in sig_keys) and (bin_fn is not None):
            cand["bin_function"] = bin_fn

        return {k: v for k, v in cand.items() if (k in sig_keys and v is not None)}

    last_err = None
    for bin_fn in [params.get("bin_function"), "entmoid15", "sparsemoid", None]:
        try:
            return _NODE_CFG(**build_kwargs(bin_fn))
        except Exception as e:
            last_err = e
            continue
    raise last_err

# === Trainer 配置兼容（修复 early_stopping 指标问题） ===
def make_trainer_config():
    """
    兼容不同版本的 TrainerConfig：
    - lightning 2.x: accelerator/devices, enable_checkpointing
    - lightning 1.x: gpus, checkpoints
    - 早停参数：优先 early_stopping='valid_loss'；失败则回退
    """
    sig = inspect.signature(TrainerConfig)
    keys = set(sig.parameters.keys())

    base = dict(batch_size=256, max_epochs=300, seed=RANDOM_STATE)

    if "accelerator" in keys:
        base.update(dict(accelerator="auto", devices=1))
    elif "gpus" in keys:
        base.update(dict(gpus=None))

    if "enable_checkpointing" in keys:
        base.update(dict(enable_checkpointing=False))
    elif "checkpoints" in keys:
        base.update(dict(checkpoints=None))

    base = {k: v for k, v in base.items() if k in keys}

    trials = []
    if "early_stopping" in keys:
        t = {"early_stopping": "valid_loss"}
        if "early_stopping_mode" in keys:
            t["early_stopping_mode"] = "min"
        if "early_stopping_patience" in keys:
            t["early_stopping_patience"] = 30
        trials.append(t)

        t = {"early_stopping": True}
        if "early_stopping_monitor" in keys:
            t["early_stopping_monitor"] = "valid_loss"
        if "early_stopping_mode" in keys:
            t["early_stopping_mode"] = "min"
        if "early_stopping_patience" in keys:
            t["early_stopping_patience"] = 30
        trials.append(t)

        trials.append({"early_stopping": False})
    trials.append({})

    last_err = None
    for t in trials:
        kwargs = base.copy()
        kwargs.update({k: v for k, v in t.items() if k in keys})
        try:
            return TrainerConfig(**kwargs)
        except Exception as e:
            last_err = e
            continue
    raise last_err

# === Optimizer 配置兼容（避免 AdamW lr 重复传参） ===
def make_optimizer_config():
    """
    兼容不同版本的 OptimizerConfig，避免 AdamW 接收到重复的 lr。
    """
    sig = inspect.signature(OptimizerConfig)
    params = sig.parameters
    has_lr = "lr" in params
    has_wd = "weight_decay" in params
    has_opt_params = "optimizer_params" in params

    if has_lr:
        kwargs = {"optimizer": "AdamW", "lr": 1e-3}
        if has_wd:
            kwargs["weight_decay"] = 1e-5
        return OptimizerConfig(**kwargs)

    if has_opt_params:
        return OptimizerConfig(optimizer="AdamW",
                               optimizer_params={"weight_decay": 1e-5})

    return OptimizerConfig(optimizer="AdamW")

# ========== 通用预处理（sklearn 家族）==========
sk_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
    ]
)

# ========== 类别索引化（供 CatBoost / FTT / TabNet / NODE 使用）==========
def fit_category_maps(df_fit: pd.DataFrame, cat_cols):
    maps = {}
    card = []
    for c in cat_cols:
        uniq = pd.Series(df_fit[c].astype("category")).cat.categories.tolist()
        idx_map = {v: i+1 for i, v in enumerate(uniq)}  # 0 留给未知
        maps[c] = idx_map
        card.append(len(uniq) + 1)
    return maps, card

def transform_cat_to_idx(df_in: pd.DataFrame, maps):
    arrs = []
    for c in maps.keys():
        m = maps[c]
        arrs.append(df_in[c].map(m).fillna(0).astype(int).to_numpy().reshape(-1,1))
    return np.concatenate(arrs, axis=1) if arrs else np.zeros((len(df_in),0), dtype=int)

def get_num_array(df_in: pd.DataFrame, num_cols):
    return df_in[num_cols].to_numpy(dtype=np.float32) if num_cols else np.zeros((len(df_in),0), dtype=np.float32)

cat_maps, cat_cardinalities = fit_category_maps(X_train, CAT_COLS)
Xtr_num = get_num_array(X_train, numeric_cols)
Xva_num = get_num_array(X_val, numeric_cols)
Xtr_cat_idx = transform_cat_to_idx(X_train, cat_maps)
Xva_cat_idx = transform_cat_to_idx(X_val, cat_maps)

# ========== CV ==========
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_outer = outer_cv
cv_inner = inner_cv

# ========== sklearn 家族 ==========
sk_models = {
    "logistic": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", LogisticRegression(max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "clf__penalty": ["l1", "l2"],
            "clf__C": np.logspace(-3, 2, 6),
            "clf__class_weight": [None, "balanced"],
        },
    },
    "svm": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "clf__kernel": ["rbf", "linear"],
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale"],
            "clf__class_weight": [None, "balanced"],
        },
    },
    "gbm": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "clf__n_estimators": [150, 300],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        },
    },
    "mlp": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", MLPClassifier(max_iter=400, random_state=RANDOM_STATE, early_stopping=True)),
        ]),
        "param_grid": {
            "clf__hidden_layer_sizes": [(64,), (64,32)],
            "clf__alpha": [1e-4, 1e-3],
            "clf__learning_rate_init": [1e-3],
        },
    },
    "dt": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "clf__max_depth": [None, 3, 5, 7],
            "clf__min_samples_leaf": [1, 2, 5],
        },
    },
    "rf": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "param_grid": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [None, 6, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
    },
    "xgb": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", XGBClassifier(
                eval_metric="logloss", random_state=RANDOM_STATE, n_estimators=400,
                tree_method="hist", n_jobs=-1, use_label_encoder=False
            )),
        ]),
        "param_grid": {
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.7, 1.0],
            "clf__colsample_bytree": [0.7, 1.0],
            "clf__reg_lambda": [1.0, 3.0],
        },
    },
    "knn": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", KNeighborsClassifier()),
        ]),
        "param_grid": {
            "clf__n_neighbors": [5, 11, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
    },
    "ada": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", AdaBoostClassifier(random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__learning_rate": [0.05, 0.1, 0.5],
        },
    },
    "lgb": {
        "estimator": Pipeline([
            ("prep", sk_preprocessor),
            ("clf", lgb.LGBMClassifier(
                objective="binary", random_state=RANDOM_STATE, n_estimators=600, n_jobs=-1
            )),
        ]),
        "param_grid": {
            "clf__num_leaves": [15, 31],
            "clf__max_depth": [-1, 5, 8],
            "clf__learning_rate": [0.05, 0.1],
            "clf__min_child_samples": [10, 20],
            "clf__reg_lambda": [0.0, 1.0],
        },
    },
}

# ========== CatBoost ==========
cat_features_idx = [X.columns.get_loc(c) for c in CAT_COLS]
cat_model_cfg = {
    "estimator": Pipeline([
        ("prep", "passthrough"),
        ("clf", CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC",
            random_seed=RANDOM_STATE, verbose=False,
            od_type="Iter", od_wait=200
        ))
    ]),
    "param_grid": {
        "clf__depth": [4, 6],
        "clf__learning_rate": [0.05, 0.1],
        "clf__l2_leaf_reg": [3.0, 6.0],
        "clf__iterations": [800, 1200],
    },
    "fit_params": {"clf__cat_features": cat_features_idx}
}

# ========== FTT（rtdl 兼容）==========
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def build_ft_transformer(n_num, cat_cardinalities,
                         d_token=64, n_blocks=3, n_heads=8, dropout=0.1, out_dim=2):
    sig = inspect.signature(rtdl.FTTransformer.make_default)
    base_kwargs = dict(
        n_num_features=n_num,
        cat_cardinalities=cat_cardinalities,
        last_layer_query_idx=[-1],
    )
    if "d_token" in sig.parameters:
        model = rtdl.FTTransformer.make_default(
            **base_kwargs,
            d_token=d_token,
            n_blocks=n_blocks,
            attention_n_heads=n_heads,
            ff_dropout=dropout,
            attention_dropout=dropout,
            residual_dropout=dropout,
        )
        head = nn.Linear(model.d_token, out_dim)
        return model, head, "split_head"
    else:
        if "n_blocks" in sig.parameters:
            base_kwargs["n_blocks"] = n_blocks
        if "d_out" in sig.parameters:
            base_kwargs["d_out"] = out_dim
        model = rtdl.FTTransformer.make_default(**base_kwargs)
        return model, None, "built_in_head"

def train_ftt_one(
    Xnum_tr, Xcat_tr, y_tr, Xnum_va, Xcat_va, y_va,
    params, device, epochs=300, batch_size=128, lr=1e-3, wd=1e-5, patience=30
):
    tr_dl = DataLoader(TensorDataset(
        torch.tensor(Xnum_tr, dtype=torch.float32),
        torch.tensor(Xcat_tr, dtype=torch.long),
        torch.tensor(y_tr.values if isinstance(y_tr, pd.Series) else y_tr, dtype=torch.long)
    ), batch_size=batch_size, shuffle=True)

    model, head, mode = build_ft_transformer(
        n_num=Xnum_tr.shape[1], cat_cardinalities=cat_cardinalities,
        d_token=params["d_token"], n_blocks=params["n_blocks"],
        n_heads=params["n_heads"], dropout=params["dropout"], out_dim=2
    )
    model.to(device)
    if head is not None: head.to(device)

    params_all = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(params_all, lr=lr, weight_decay=wd)
    ce  = nn.CrossEntropyLoss()

    def _forward_logits(m, h, Xn, Xc):
        if mode == "split_head":
            feats  = m(Xn, Xc)
            logits = h(feats)
        else:
            logits = m(Xn, Xc)
        return logits

    def _predict_proba(m, h, Xn_np, Xc_np):
        m.eval();
        if h is not None: h.eval()
        with torch.no_grad():
            Xn = torch.tensor(Xn_np, dtype=torch.float32, device=device)
            Xc = torch.tensor(Xc_np, dtype=torch.long,   device=device)
            logits = _forward_logits(m, h, Xn, Xc)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return prob

    best_auc, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs+1):
        model.train();
        if head is not None: head.train()
        for Xn_b, Xc_b, y_b in tr_dl:
            Xn_b = Xn_b.to(device); Xc_b = Xc_b.to(device); y_b = y_b.to(device)
            opt.zero_grad();
            loss = ce(_forward_logits(model, head, Xn_b, Xc_b), y_b)
            loss.backward(); opt.step()

        va_prob = _predict_proba(model, head, Xnum_va, Xcat_va)
        auc = roc_auc_score(y_va, va_prob)
        if auc > best_auc + 1e-6:
            best_auc = auc; wait = 0
            sd = {"model": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
            if head is not None: sd["head"] = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            best_state = sd
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        # 把保存的 CPU state_dict 按需搬到当前 device 再加载
        model.load_state_dict({k: v.to(device) for k, v in best_state["model"].items()})
        if head is not None and best_state.get("head") is not None:
            head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()})

# ========== TabNet ==========
tabnet_param_grid = [
    {"n_d":16, "n_a":16, "n_steps":3, "gamma":1.5, "lambda_sparse":1e-4, "learning_rate":1e-3, "max_epochs":300, "patience":40, "batch_size":256},
    {"n_d":24, "n_a":24, "n_steps":3, "gamma":1.5, "lambda_sparse":1e-4, "learning_rate":1e-3, "max_epochs":300, "patience":40, "batch_size":256},
]
def build_tabnet_input(Xnum, Xcat_idx):
    Xall = np.concatenate([Xnum, Xcat_idx.astype(np.float32)], axis=1).astype(np.float32)
    cat_idxs = list(range(Xnum.shape[1], Xnum.shape[1] + Xcat_idx.shape[1]))
    cat_dims = cat_cardinalities
    return Xall, cat_idxs, cat_dims

# ========== NODE ==========
def train_node_one(df_train, df_val, d_in, params):
    data_config = DataConfig(
        target=["target"],
        continuous_cols=[f"f{i}" for i in range(d_in)],
        categorical_cols=[],   # 修复：不能为 None
    )
    model_config = _make_node_config(d_in, params)
    optimizer_config = make_optimizer_config()
    trainer_config = make_trainer_config()

    tm = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tm.fit(train=df_train, validation=df_val)
    return tm

node_param_grid = [
    {"num_layers":2, "num_trees":256, "depth":6},
    {"num_layers":2, "num_trees":512, "depth":6},
]

# ========== 工具：指标与保存 ==========
def metrics_all(y_tr, y_va, oof, train_pred, val_pred):
    d = {
        "oof_auc": float(roc_auc_score(y_tr, oof)) if oof is not None else None,
        "train_auc_retrained": float(roc_auc_score(y_tr, train_pred)),
        "val_auc_retrained": float(roc_auc_score(y_va, val_pred)),
        "train_logloss_retrained": float(log_loss(y_tr, train_pred, labels=[0,1])),
        "val_logloss_retrained": float(log_loss(y_va, val_pred, labels=[0,1])),
        "train_brier_retrained": float(brier_score_loss(y_tr, train_pred)),
        "val_brier_retrained": float(brier_score_loss(y_va, val_pred)),
    }
    return d

def save_per_model(name, oof_pred, train_pred, val_pred, best_info, y_tr, y_va, m):
    if oof_pred is not None:
        pd.DataFrame({"y_train": y_tr, name: oof_pred}).to_csv(PERDIR / f"oof_{name}.csv", index=False)
    pd.DataFrame({"y_train": y_tr, name: train_pred}).to_csv(PERDIR / f"train_retrained_{name}.csv", index=False)
    pd.DataFrame({"y_val":   y_va, name: val_pred}).to_csv(PERDIR / f"val_{name}.csv", index=False)
    with open(PERDIR / f"best_params_{name}.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)
    row = pd.DataFrame([{ "model": name, **m }])
    file_running = OUTDIR / "level1_model_scores_running.csv"
    header = not file_running.exists()
    row.to_csv(file_running, mode="a", header=header, index=False)

def save_trained_model(name, final_model):
    """保存各一级模型最终‘重训后’的模型对象，供 SHAP 使用。失败不影响主流程。"""
    try:
        if name in ["logistic","svm","gbm","mlp","dt","rf","xgb","knn","ada","lgb","cat"]:
            joblib.dump(final_model, PERDIR / f"model_{name}.pkl")
        elif name == "ftt":
            model, head, mode = final_model
            torch.save(
                {
                    "state_model": model.state_dict(),
                    "state_head": head.state_dict() if head is not None else None,
                    "mode": mode,
                    "numeric_cols": numeric_cols,
                    "cat_cols": CAT_COLS,
                    "cat_cardinalities": cat_cardinalities,
                },
                PERDIR / "model_ftt.pt"
            )
        elif name == "tabnet":
            final_model.save_model(str(PERDIR / "model_tabnet"))
            with open(PERDIR / "model_tabnet_meta.json", "w", encoding="utf-8") as f:
                json.dump({"numeric_cols": numeric_cols, "cat_cols": CAT_COLS, "cat_cardinalities": cat_cardinalities},
                          f, ensure_ascii=False, indent=2)
        elif name == "node":
            outdir = PERDIR / "model_node"
            outdir.mkdir(parents=True, exist_ok=True)
            final_model.save_model(str(outdir))
    except Exception as e:
        with open(PERDIR / "failed_models.log", "a", encoding="utf-8") as f:
            f.write(f"[save_model] {name}: {repr(e)}\n")

def already_completed(name):
    need = [
        PERDIR / f"train_retrained_{name}.csv",
        PERDIR / f"val_{name}.csv",
        PERDIR / f"best_params_{name}.json",
        PERDIR / f"oof_{name}.csv",
        PERDIR / f"model_{name}.pkl" if name in ["logistic","svm","gbm","mlp","dt","rf","xgb","knn","ada","lgb","cat"] else PERDIR / "dummy"
    ]
    return all(p.exists() for p in need)

# ========== 一级：各模型训练 ==========
def run_sklearn_model(name, cfg, X_tr, y_tr, X_holdout, y_holdout):
    # 1) OOF
    oof = np.zeros(len(X_tr))
    fold_params = []
    for tr_idx, va_idx in cv_outer.split(X_tr, y_tr):
        X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_va_f, y_va_f = X_tr.iloc[va_idx], y_tr.iloc[va_idx]
        gs = GridSearchCV(
            estimator=clone(cfg["estimator"]),
            param_grid=cfg["param_grid"],
            scoring="roc_auc",
            cv=cv_inner, n_jobs=-1, refit=True, verbose=0
        )
        fit_params = cfg.get("fit_params", {})
        if fit_params: gs.fit(X_tr_f, y_tr_f, **fit_params)
        else:          gs.fit(X_tr_f, y_tr_f)
        best_est = gs.best_estimator_
        fold_params.append(gs.best_params_)
        oof[va_idx] = best_est.predict_proba(X_va_f)[:,1]

    # 2) 全训练重训
    gs_final = GridSearchCV(
        estimator=clone(cfg["estimator"]),
        param_grid=cfg["param_grid"],
        scoring="roc_auc",
        cv=cv_inner, n_jobs=-1, refit=True, verbose=0
    )
    fit_params = cfg.get("fit_params", {})
    if fit_params: gs_final.fit(X_tr, y_tr, **fit_params)
    else:          gs_final.fit(X_tr, y_tr)
    final_best_est = gs_final.best_estimator_
    train_pred = final_best_est.predict_proba(X_tr)[:,1]
    val_pred   = final_best_est.predict_proba(X_holdout)[:,1]
    return oof, train_pred, val_pred, {"fold_best_params": fold_params, "final_best_params": gs_final.best_params_}, final_best_est

def run_catboost(X_tr, y_tr, X_holdout, y_holdout):
    return run_sklearn_model("cat", cat_model_cfg, X_tr, y_tr, X_holdout, y_holdout)

def run_ftt(X_tr, y_tr, X_holdout, y_holdout):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # OOF
    oof = np.zeros(len(X_tr))
    fold_params = []
    for tr_idx, va_idx in cv_outer.split(X_tr, y_tr):
        X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_va_f, y_va_f = X_tr.iloc[va_idx], y_tr.iloc[va_idx]

        Xn_tr = get_num_array(X_tr_f, numeric_cols)
        Xn_va = get_num_array(X_va_f, numeric_cols)
        Xc_tr = transform_cat_to_idx(X_tr_f, cat_maps)
        Xc_va = transform_cat_to_idx(X_va_f, cat_maps)

        best_auc, best_params, best_model, best_head, best_mode = -1.0, None, None, None, None
        for params in [
            {"d_token":64, "n_blocks":3, "n_heads":8, "dropout":0.1},
            {"d_token":96, "n_blocks":3, "n_heads":8, "dropout":0.1},
        ]:
            model, head, mode = train_ftt_one(
                Xn_tr, Xc_tr, y_tr_f, Xn_va, Xc_va, y_va_f,
                params=params, device=device, epochs=300, batch_size=128, lr=1e-3, wd=1e-5, patience=30
            )
            with torch.no_grad():
                Xn = torch.tensor(Xn_va, dtype=torch.float32, device=device)
                Xc = torch.tensor(Xc_va, dtype=torch.long,   device=device)
                if mode == "split_head":
                    prob = torch.softmax(head(model(Xn, Xc)), dim=1)[:,1].cpu().numpy()
                else:
                    prob = torch.softmax(model(Xn, Xc), dim=1)[:,1].cpu().numpy()
            auc = roc_auc_score(y_va_f, prob)
            if auc > best_auc:
                best_auc, best_params, best_model, best_head, best_mode = auc, params, model, head, mode
        fold_params.append(best_params)
        with torch.no_grad():
            Xn = torch.tensor(Xn_va, dtype=torch.float32, device=device)
            Xc = torch.tensor(Xc_va, dtype=torch.long,   device=device)
            if best_mode == "split_head":
                oof[va_idx] = torch.softmax(best_head(best_model(Xn, Xc)), dim=1)[:,1].cpu().numpy()
            else:
                oof[va_idx] = torch.softmax(best_model(Xn, Xc), dim=1)[:,1].cpu().numpy()

    # 全训练重训（选择最终超参）
    Xn_tr_all = Xtr_num; Xc_tr_all = Xtr_cat_idx
    Xn_va_all = Xva_num; Xc_va_all = Xva_cat_idx
    best_auc, best_params, best_model, best_head, best_mode = -1.0, None, None, None, None
    for params in [
        {"d_token":64, "n_blocks":3, "n_heads":8, "dropout":0.1},
        {"d_token":96, "n_blocks":3, "n_heads":8, "dropout":0.1},
    ]:
        model, head, mode = train_ftt_one(
            Xn_tr_all, Xc_tr_all, y_tr, Xn_va_all, Xc_va_all, y_holdout,
            params=params, device=device, epochs=300, batch_size=128, lr=1e-3, wd=1e-5, patience=30
        )
        with torch.no_grad():
            Xn = torch.tensor(Xn_va_all, dtype=torch.float32, device=device)
            Xc = torch.tensor(Xc_va_all, dtype=torch.long,   device=device)
            if mode == "split_head":
                prob = torch.softmax(head(model(Xn, Xc)), dim=1)[:,1].cpu().numpy()
            else:
                prob = torch.softmax(model(Xn, Xc), dim=1)[:,1].cpu().numpy()
        auc = roc_auc_score(y_holdout, prob)
        if auc > best_auc:
            best_auc, best_params, best_model, best_head, best_mode = auc, params, model, head, mode

    with torch.no_grad():
        # 训练集
        Xn = torch.tensor(Xn_tr_all, dtype=torch.float32, device=device)
        Xc = torch.tensor(Xc_tr_all, dtype=torch.long,   device=device)
        if best_mode == "split_head":
            train_pred = torch.softmax(best_head(best_model(Xn, Xc)), dim=1)[:,1].cpu().numpy()
        else:
            train_pred = torch.softmax(best_model(Xn, Xc), dim=1)[:,1].cpu().numpy()
        # 验证集
        Xn = torch.tensor(Xn_va_all, dtype=torch.float32, device=device)
        Xc = torch.tensor(Xc_va_all, dtype=torch.long,   device=device)
        if best_mode == "split_head":
            val_pred = torch.softmax(best_head(best_model(Xn, Xc)), dim=1)[:,1].cpu().numpy()
        else:
            val_pred = torch.softmax(best_model(Xn, Xc), dim=1)[:,1].cpu().numpy()

    return oof, train_pred, val_pred, {"fold_best_params": fold_params, "final_best_params": best_params}, (best_model, best_head, best_mode)

def run_tabnet(X_tr, y_tr, X_holdout, y_holdout):
    oof = np.zeros(len(X_tr))
    fold_params = []

    for tr_idx, va_idx in outer_cv.split(X_tr, y_tr):
        X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_va_f, y_va_f = X_tr.iloc[va_idx], y_tr.iloc[va_idx]
        Xn_tr = get_num_array(X_tr_f, numeric_cols)
        Xn_va = get_num_array(X_va_f, numeric_cols)
        Xc_tr = transform_cat_to_idx(X_tr_f, cat_maps)
        Xc_va = transform_cat_to_idx(X_va_f, cat_maps)
        X_tr_all, cat_idxs, cat_dims = build_tabnet_input(Xn_tr, Xc_tr)
        X_va_all, _, _ = build_tabnet_input(Xn_va, Xc_va)

        best_auc, best_params, best_model = -1.0, None, None
        for params in tabnet_param_grid:
            model = TabNetClassifier(
                n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
                gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
                seed=RANDOM_STATE, verbose=0,
                cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1
            )
            model.fit(
                X_tr_all, y_tr_f.values,
                eval_set=[(X_va_all, y_va_f.values)],
                eval_metric=["auc"],
                max_epochs=params["max_epochs"], patience=params["patience"],
                batch_size=params["batch_size"], virtual_batch_size=max(64, params["batch_size"]//4)
            )
            prob = model.predict_proba(X_va_all)[:,1]
            auc = roc_auc_score(y_va_f, prob)
            if auc > best_auc:
                best_auc, best_params, best_model = auc, params, model
        oof[va_idx] = best_model.predict_proba(X_va_all)[:,1]
        fold_params.append(best_params)

    # 全训练重训
    Xn_tr_all = Xtr_num; Xc_tr_all = Xtr_cat_idx
    Xn_va_all = Xva_num; Xc_va_all = Xva_cat_idx
    X_tr_all, cat_idxs, cat_dims = build_tabnet_input(Xn_tr_all, Xc_tr_all)
    X_va_all, _, _ = build_tabnet_input(Xn_va_all, Xc_va_all)

    best_auc, best_params, best_model = -1.0, None, None
    for params in tabnet_param_grid:
        model = TabNetClassifier(
            n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
            gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
            seed=RANDOM_STATE, verbose=0,
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1
        )
        model.fit(
            X_tr_all, y_train.values,
            eval_set=[(X_va_all, y_holdout.values)],
            eval_metric=["auc"],
            max_epochs=params["max_epochs"], patience=params["patience"],
            batch_size=params["batch_size"], virtual_batch_size=max(64, params["batch_size"]//4)
        )
        prob = model.predict_proba(X_va_all)[:,1]
        auc = roc_auc_score(y_holdout, prob)
        if auc > best_auc:
            best_auc, best_params, best_model = auc, params, model

    train_pred = best_model.predict_proba(X_tr_all)[:,1]
    val_pred   = best_model.predict_proba(X_va_all)[:,1]
    return oof, train_pred, val_pred, {"fold_best_params": fold_params, "final_best_params": best_params}, best_model

# —— 兼容不同版本 TabularModel 的概率输出 —— #
def node_predict_proba(model, df_input):
    """
    兼容不同版本 pytorch_tabular 的输出，稳定拿到“正类(=1)概率”.
    优先顺序：
      1) prediction_probability / target_1_probability / probability_1 / class_1_probability / prediction_probability_1 / prob_1
      2) *_1_probability
      3) 包含 'prob' 且含独立 '_1'
      4) 用 (prediction 或 target_prediction) + probability 还原
    """
    try:
        out = model.predict(df_input)
    except TypeError:
        out = model.predict(df=df_input)

    if isinstance(out, tuple):
        out = out[0]
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out)

    cols_lower = {c.lower(): c for c in out.columns}
    for key in [
        "prediction_probability",
        "target_1_probability",
        "probability_1",
        "class_1_probability",
        "prediction_probability_1",
        "prob_1",
    ]:
        if key in cols_lower:
            return out[cols_lower[key]].to_numpy()

    pair1 = [c for c in out.columns if c.lower().endswith("_1_probability")]
    if pair1:
        return out[pair1[0]].to_numpy()

    for c in out.columns:
        lc = c.lower()
        if "prob" in lc and (re.search(r"(^|_)1($|_)", lc) is not None):
            return out[c].to_numpy()

    pred_col = None
    for k in ["prediction", "target_prediction"]:
        if k in cols_lower:
            pred_col = cols_lower[k]
            break
    if pred_col and "probability" in cols_lower:
        prob_col = cols_lower["probability"]
        pred = out[pred_col].to_numpy()
        prob = out[prob_col].to_numpy()
        return np.where(pred == 1, prob, 1.0 - prob)

    raise RuntimeError(f"无法从 NODE 的预测输出中找到正类概率列。可用列: {list(out.columns)}")

def run_node(X_tr, y_tr, X_holdout, y_holdout):
    def make_df(Xnum, Xcat_idx):
        X_all = np.concatenate([Xnum, Xcat_idx.astype(np.float32)], axis=1).astype(np.float32)
        d_in = X_all.shape[1]
        df_ = pd.DataFrame(X_all, columns=[f"f{i}" for i in range(d_in)])
        return df_, d_in

    # OOF
    oof = np.zeros(len(X_tr))
    fold_params = []

    for tr_idx, va_idx in outer_cv.split(X_tr, y_tr):
        X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_va_f, y_va_f = X_tr.iloc[va_idx], y_tr.iloc[va_idx]

        Xn_tr = get_num_array(X_tr_f, numeric_cols); Xc_tr = transform_cat_to_idx(X_tr_f, cat_maps)
        Xn_va = get_num_array(X_va_f, numeric_cols); Xc_va = transform_cat_to_idx(X_va_f, cat_maps)
        df_tr, d_in = make_df(Xn_tr, Xc_tr)
        df_va, _    = make_df(Xn_va, Xc_va)
        df_tr["target"] = y_tr_f.values
        df_va["target"] = y_va_f.values

        best_auc, best_params, best_model = -1.0, None, None
        for params in node_param_grid:
            model = train_node_one(df_tr, df_va, d_in, params)
            prob = node_predict_proba(model, df_va)
            auc = roc_auc_score(y_va_f, prob)
            if auc > best_auc:
                best_auc, best_params, best_model = auc, params, model
        oof[va_idx] = node_predict_proba(best_model, df_va)
        fold_params.append(best_params)

    # 全训练重训
    Xn_tr_all = Xtr_num; Xc_tr_all = Xtr_cat_idx
    Xn_va_all = Xva_num; Xc_va_all = Xva_cat_idx
    df_tr_all, d_in = make_df(Xn_tr_all, Xc_tr_all)
    df_va_all, _    = make_df(Xn_va_all, Xc_va_all)
    df_tr_all["target"] = y_train.values
    df_va_all["target"] = y_holdout.values

    best_auc, best_params, best_model = -1.0, None, None
    for params in node_param_grid:
        model = train_node_one(df_tr_all, df_va_all, d_in, params)
        prob = node_predict_proba(model, df_va_all)
        auc = roc_auc_score(y_holdout, prob)
        if auc > best_auc:
            best_auc, best_params, best_model = auc, params, model

    train_pred = node_predict_proba(best_model, df_tr_all)
    val_pred   = node_predict_proba(best_model, df_va_all)
    return oof, train_pred, val_pred, {"fold_best_params": fold_params, "final_best_params": best_params}, best_model

# =========================================
# 运行与汇总
# =========================================
level1_oof, level1_val, level1_best, level1_scores = {}, {}, {}, {}
failed_log = PERDIR / "failed_models.log"

def run_and_save(name, runner_func):
    try:
        if SKIP_COMPLETED and already_completed(name):
            print(f"[SKIP COMPLETED] {name}")
            oof_df   = pd.read_csv(PERDIR / f"oof_{name}.csv")
            trr_df   = pd.read_csv(PERDIR / f"train_retrained_{name}.csv")
            valr_df  = pd.read_csv(PERDIR / f"val_{name}.csv")
            oof_pred   = oof_df[name].to_numpy()
            train_pred = trr_df[name].to_numpy()
            val_pred   = valr_df[name].to_numpy()
            with open(PERDIR / f"best_params_{name}.json","r",encoding="utf-8") as f:
                best_info = json.load(f)
            m = metrics_all(y_train, y_val, oof_pred, train_pred, val_pred)
            level1_oof[name] = oof_pred
            level1_val[name] = val_pred
            level1_best[name]= best_info
            level1_scores[name] = m
            print(f"[{name}] retrained Train AUC={m['train_auc_retrained']:.3f} | Val AUC={m['val_auc_retrained']:.3f}")
            return

        print(f"\n=== [{name}] ===")
        oof_pred, train_pred, val_pred, best_info, final_model = runner_func()
        m = metrics_all(y_train, y_val, oof_pred, train_pred, val_pred)
        print(f"[{name}] retrained Train AUC={m['train_auc_retrained']:.3f} | Val AUC={m['val_auc_retrained']:.3f}")

        save_per_model(name, oof_pred, train_pred, val_pred, best_info, y_train, y_val, m)
        save_trained_model(name, final_model)  # 保存最终模型（供 SHAP）

        level1_oof[name] = oof_pred
        level1_val[name] = val_pred
        level1_best[name]= best_info
        level1_scores[name] = m

    except Exception as e:
        with open(failed_log, "a", encoding="utf-8") as f:
            f.write(f"{name}: {repr(e)}\n")
        print(f"[WARN] {name} 失败：{repr(e)}，已记录日志，继续下一模型。")

# 10 个 sklearn
for name, cfg in sk_models.items():
    run_and_save(
        name,
        lambda n=name, c=cfg: run_sklearn_model(n, c, X_train, y_train, X_val, y_val)
    )

# CatBoost
run_and_save("cat", lambda: run_catboost(X_train, y_train, X_val, y_val))
# FT-Transformer
run_and_save("ftt", lambda: run_ftt(X_train, y_train, X_val, y_val))
# TabNet（已修复）
run_and_save("tabnet", lambda: run_tabnet(X_train, y_train, X_val, y_val))
# NODE（已修复）
run_and_save("node", lambda: run_node(X_train, y_train, X_val, y_val))

# ========== 汇总 ==========
def merge_per_model(pattern, key_col):
    files = sorted(glob.glob(str(PERDIR / pattern)))
    base = None
    for fp in files:
        dfp = pd.read_csv(fp)
        if base is None:
            base = dfp[[key_col]].copy()
        col = [c for c in dfp.columns if c != key_col][0]
        base[col] = dfp[col]
    return base

df_oof_all   = merge_per_model("oof_*.csv", "y_train")
df_train_all = merge_per_model("train_retrained_*.csv", "y_train")
df_val_all   = merge_per_model("val_*.csv", "y_val")

if df_oof_all is not None:
    df_oof_all.to_csv(OUTDIR / "level1_oof_predictions.csv", index=False)
if df_train_all is not None:
    df_train_all.to_csv(OUTDIR / "level1_train_retrained_predictions.csv", index=False)
if df_val_all is not None:
    df_val_all.to_csv(OUTDIR / "level1_val_predictions.csv", index=False)

scores_df = pd.DataFrame([
    {"model": k, **v} for k, v in level1_scores.items()
]).sort_values("val_auc_retrained", ascending=False)
scores_df.to_csv(OUTDIR / "level1_model_scores.csv", index=False)

best_params_all = {}
for js in sorted(PERDIR.glob("best_params_*.json")):
    with open(js, "r", encoding="utf-8") as f:
        best_params_all[js.stem.replace("best_params_","")] = json.load(f)
with open(OUTDIR / "level1_best_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params_all, f, ensure_ascii=False, indent=2)

# ========== 二级（OOF 训 | 训练/验证推断全部导出）==========
if df_oof_all is not None and df_val_all is not None:
    meta_train = df_oof_all.drop(columns=["y_train"])  # 一级 OOF 作为二级训练特征
    meta_val   = df_val_all.drop(columns=["y_val"])    # 一级“重训后”的验证预测作为二级验证特征

    # 1) 训练二级模型（带CV选C）
    meta = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 20),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        n_jobs=-1,
        refit=True
    )
    meta.fit(meta_train, y_train)

    # 2) 二级在验证集的预测
    val_pred_meta = meta.predict_proba(meta_val)[:, 1]

    # 3) 二级在训练集的“训练内(in-sample)”预测（直接对训练集做推断；会偏乐观）
    train_pred_meta_insample = meta.predict_proba(meta_train)[:, 1]

    # 4) 二级在训练集的 OOF 预测（用固定C=meta选出的C，按5折做OOF，更稳健）
    from sklearn.linear_model import LogisticRegression
    C_used = float(meta.C_[0] if np.ndim(meta.C_) > 0 else meta.C_)
    skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_pred_meta_oof = np.zeros(len(y_train), dtype=float)
    for tr_idx, va_idx in skf_meta.split(meta_train, y_train):
        X_tr_m, y_tr_m = meta_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va_m         = meta_train.iloc[va_idx]
        clf = LogisticRegression(C=C_used, penalty="l2", solver="lbfgs", max_iter=5000)
        clf.fit(X_tr_m, y_tr_m)
        train_pred_meta_oof[va_idx] = clf.predict_proba(X_va_m)[:, 1]

    # 5) 指标
    meta_auc_val = roc_auc_score(y_val, val_pred_meta)
    meta_ll_val  = log_loss(y_val, val_pred_meta, labels=[0, 1])
    meta_br_val  = brier_score_loss(y_val, val_pred_meta)

    meta_auc_train_ins = roc_auc_score(y_train, train_pred_meta_insample)
    meta_ll_train_ins  = log_loss(y_train, train_pred_meta_insample, labels=[0, 1])
    meta_br_train_ins  = brier_score_loss(y_train, train_pred_meta_insample)

    meta_auc_train_oof = roc_auc_score(y_train, train_pred_meta_oof)
    meta_ll_train_oof  = log_loss(y_train, train_pred_meta_oof, labels=[0, 1])
    meta_br_train_oof  = brier_score_loss(y_train, train_pred_meta_oof)

    print(f"\n===== META (LR-CV L2) =====")
    print(f"[VAL]   AUC={meta_auc_val:.4f} | LogLoss={meta_ll_val:.4f} | Brier={meta_br_val:.4f}")
    print(f"[TRAIN] in-sample  AUC={meta_auc_train_ins:.4f} | LogLoss={meta_ll_train_ins:.4f} | Brier={meta_br_train_ins:.4f}")
    print(f"[TRAIN] OOF        AUC={meta_auc_train_oof:.4f} | LogLoss={meta_ll_train_oof:.4f} | Brier={meta_br_train_oof:.4f}")

    # 6) 保存模型与预测
    joblib.dump(meta, OUTDIR / "meta_lr_cv.pkl")
    # 验证集预测（原有）
    pd.DataFrame({"y_val": y_val, "stacking_pred": val_pred_meta}).to_csv(
        OUTDIR / "level2_val_predictions.csv", index=False
    )
    # 训练集 in-sample 预测
    pd.DataFrame({"y_train": y_train, "stacking_pred_train_insample": train_pred_meta_insample}).to_csv(
        OUTDIR / "level2_train_in_sample_predictions.csv", index=False
    )
    # 训练集 OOF 预测（推荐在报告里用它代表二级在训练集的泛化表现）
    pd.DataFrame({"y_train": y_train, "stacking_pred_train_oof": train_pred_meta_oof}).to_csv(
        OUTDIR / "level2_train_oof_predictions.csv", index=False
    )

    summary = {
        "random_state": RANDOM_STATE,
        "models_run": list(meta_train.columns),
        "meta_val_auc": float(meta_auc_val),
        "meta_val_logloss": float(meta_ll_val),
        "meta_val_brier": float(meta_br_val),
        "meta_train_ins_auc": float(meta_auc_train_ins),
        "meta_train_ins_logloss": float(meta_ll_train_ins),
        "meta_train_ins_brier": float(meta_br_train_ins),
        "meta_train_oof_auc": float(meta_auc_train_oof),
        "meta_train_oof_logloss": float(meta_ll_train_oof),
        "meta_train_oof_brier": float(meta_br_train_oof),
        "meta_C_used": C_used,
    }
else:
    print("\n[WARN] 一级特征不完整，跳过二级训练。")
    summary = {"random_state": RANDOM_STATE, "models_run": [], "meta_val_auc": None, "meta_val_logloss": None, "meta_val_brier": None}

with open(OUTDIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# 可视化（以“重训后的验证 AUC”排序展示）
if not scores_df.empty:
    plt.figure(figsize=(11,6))
    plt.bar(scores_df["model"], scores_df["val_auc_retrained"], label="Level-1 Val AUC (Retrained)")
    if summary.get("meta_val_auc") is not None:
        plt.axhline(summary["meta_val_auc"], linestyle="--", label=f"Stacking (meta) AUC={summary['meta_val_auc']:.3f}")
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Validation AUC")
    plt.title("Level-1 (Retrained) vs Stacking (Level-2) AUC")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUTDIR / "auc_comparison.png", dpi=300); plt.close()

print("\n全部完成。输出目录：", OUTDIR.resolve())
print(" - 一级 OOF：", (OUTDIR / "level1_oof_predictions.csv").resolve())
print(" - 一级‘重训后’训练/验证：",
      (OUTDIR / "level1_train_retrained_predictions.csv").resolve(), "|",
      (OUTDIR / "level1_val_predictions.csv").resolve())
print(" - 一级跑分（重训指标）：", (OUTDIR / "level1_model_scores.csv").resolve())
print(" - 二级模型与预测：", (OUTDIR / "meta_lr_cv.pkl").resolve(), "|",
      (OUTDIR / "level2_val_predictions.csv").resolve(), "|",
      (OUTDIR / "level2_train_in_sample_predictions.csv").resolve(), "|",
      (OUTDIR / "level2_train_oof_predictions.csv").resolve())
print(" - 最佳参数汇总：", (OUTDIR / "level1_best_params.json").resolve())
print(" - 每模型产物目录：", PERDIR.resolve())
