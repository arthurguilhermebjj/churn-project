import argparse
import json
import pickle
import pandas as pd
import numpy as np

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV de entrada com os clientes")
    parser.add_argument("--output", required=True, help="CSV de saída com probabilidade e segmento")
    parser.add_argument("--model", default="models/churn_model.pkl", help="Caminho do modelo pickle")
    parser.add_argument("--config", default="models/model_config.json", help="Caminho do config json")
    args = parser.parse_args()

    model = load_pickle(args.model)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    threshold = float(config["threshold"])

    df = pd.read_csv(args.input)

    # prevê probabilidade
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= threshold).astype(int)

    # segmentação
    # Críticos > 0.70 | Risco Médio 0.30–0.70 | Estáveis < 0.30
    seg = np.where(proba > 0.70, "Críticos",
          np.where(proba >= 0.30, "Risco Médio", "Estáveis"))

    out = df.copy()
    out["churn_probability"] = proba
    out["churn_pred"] = pred
    out["risk_segment"] = seg

    out.to_csv(args.output, index=False)
    print("Arquivo gerado:", args.output)

if __name__ == "__main__":
    main()