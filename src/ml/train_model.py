import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    # 1) Carrega dados de treino
    #    Você teria algo como: n_codes, n_events, next_action
    df = pd.read_csv("src/ml/train_data.csv")  # Ajuste o caminho se preciso

    # 2) Separa em X (features) e y (target)
    X = df[["n_codes", "n_events"]]
    y = df["next_action"]

    # 3) Cria e treina o modelo
    model = RandomForestClassifier()
    model.fit(X, y)

    # 4) Salva o modelo em um arquivo pkl
    joblib.dump(model, "src/ml/model.pkl")
    print("Modelo treinado e salvo em src/ml/model.pkl.")

if __name__ == "__main__":
    main()
