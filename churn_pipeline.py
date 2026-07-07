import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

RAW_PATH = "data/raw/synthetic_customer_churn_dataset.csv"
THRESHOLD = 0.30
RANDOM_STATE = 42

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_total_spent(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask_invalid = df["total_spent"].astype(str).str.count(r"\.") > 1

    df.loc[~mask_invalid, "total_spent"] = pd.to_numeric(
        df.loc[~mask_invalid, "total_spent"],
        errors="coerce"
    )
    df.loc[mask_invalid, "total_spent"] = np.nan

    df = df.dropna(subset=["total_spent"]).copy()
    df["total_spent"] = df["total_spent"].astype(float)
    return df

def split_data(df: pd.DataFrame, target="churn", random_state=RANDOM_STATE):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/outputs", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1) carregar + limpar
    df_raw = load_data(RAW_PATH)
    df_clean = clean_total_spent(df_raw)
    df_clean.to_csv("data/processed/cleaned_dataset.csv", index=False)

    # 2) split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    # 3) pipeline final (LogReg balanced)
    preprocessor = build_preprocessor(X_train)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    pipeline.fit(X_train, y_train)

    # 4) avaliação final no TESTE (só agora)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    auc = roc_auc_score(y_test, y_proba_test)

    metrics_df = pd.DataFrame([{
        "threshold": THRESHOLD,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc
    }])
    metrics_df.to_csv("data/outputs/test_metrics.csv", index=False)

    print("\nMétricas no teste:")
    print(metrics_df)

    # 5) matriz de confusão
    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(cm, index=["real_0", "real_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv("data/outputs/confusion_matrix_test.csv", index=True)

    plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.title("Matriz de Confusão (Teste)")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Real 0", "Real 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix_test.png")
    plt.close()

    # 6) ROC
    fpr, tpr, thr = roc_curve(y_test, y_proba_test)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thr})
    roc_df.to_csv("data/outputs/roc_auc_test.csv", index=False)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (Teste) - AUC={auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig("images/roc_curve_test.png")
    plt.close()

    # 7) “importância” para LogReg: coeficientes (aproximação)
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(
        X_train.select_dtypes(include=["object"]).columns.tolist()
    )
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = num_features + list(cat_features)

    coefs = pipeline.named_steps["classifier"].coef_[0]
    fi = pd.DataFrame({"feature": feature_names, "coef": coefs})
    fi["abs_coef"] = fi["coef"].abs()
    fi = fi.sort_values("abs_coef", ascending=False)
    fi.to_csv("data/outputs/feature_importance_logreg.csv", index=False)

    # 8) salvar modelo + config
    save_pickle(pipeline, "models/churn_model.pkl")

    config = {
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "model_type": "LogisticRegression(class_weight=balanced)",
        "notes": "Pipeline com ColumnTransformer + OneHot + StandardScaler"
    }
    with open("models/model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nModelo salvo em models/churn_model.pkl")
    print("Config salvo em models/model_config.json")
    print("Imagens salvas em images/ e outputs em data/outputs/")

if __name__ == "__main__":
    main()