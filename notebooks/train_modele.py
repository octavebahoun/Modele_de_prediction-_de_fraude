import sys
import os

# Ajouter la racine du projet au chemin de recherche Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import load_data, prepare_data
from src.model_utils import train_forest, save_model
from src.viz_utils import setup_output_dir, plot_feature_importance, plot_correlation_matrix, plot_confusion_matrix

def main():
    # Chemins
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, "data/dataset-1.csv")
    model_path = os.path.join(base_dir, "models/random_forest_v1.joblib")
    output_dir = os.path.join(base_dir, "outputs")
    
    # 1. Chargement et préparation
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # 2. Entraînement
    model = train_forest(X_train, y_train)
    
    # 3. Sauvegarde
    save_model(model, model_path)
    
    # 4. Visualisation
    setup_output_dir(output_dir)
    plot_feature_importance(model, X_train.columns, os.path.join(output_dir, "feature_importance.png"))
    plot_correlation_matrix(df, os.path.join(output_dir, "correlation_matrix.png"))
    
    # 5. Évaluation & Matrice de confusion
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, os.path.join(output_dir, "confusion_matrix.png"))
    
    # 6. Analyse de cas concret
    indices_fraudes_reelles = y_test[y_test == 1].index
    vrais_positifs = [i for i in indices_fraudes_reelles if y_pred[list(y_test.index).index(i)] == 1]
    
    print("\n--- Analyse ---")
    if vrais_positifs:
        print(f"🔎 Cas détecté (Vrai Positif) - Index: {vrais_positifs[0]}")
    print("✨ Tout est terminé avec succès !")

if __name__ == "__main__":
    main()
