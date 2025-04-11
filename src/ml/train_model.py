import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    df = pd.read_csv("src/ml/train_data.csv")
    X = df[["n_codes", "n_events"]]
    y = df["next_action"]

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, "src/ml/model.pkl")
    print("Modelo treinado e salvo em src/ml/model.pkl.")

if __name__ == "__main__":
    main()
