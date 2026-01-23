import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def train_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un modèle Random Forest."""
    print(f"🌲 Entraînement de la forêt ({n_estimators} arbres)...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné !")
    return model

def save_model(model, path):
    """Sauvegarde le modèle sur le disque."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Modèle sauvegardé sous : {path}")

def load_model(path):
    """Charge un modèle sauvegardé."""
    return joblib.load(path)
