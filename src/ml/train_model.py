import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def main():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "train_data.csv"))
    X = df[["n_codes", "n_events"]]
    y = df["next_action"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    output_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(model, output_path)
    print(f"Modelo treinado e salvo em {output_path}")

if __name__ == "__main__":
    main()
