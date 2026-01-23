import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Charge le dataset CSV."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Données chargées avec succès depuis {file_path}")
        return df
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {file_path} est introuvable.")
        raise

def prepare_data(df, target_col='is_fraud', test_size=0.2, random_state=42):
    """Prépare les données pour l'entraînement."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
