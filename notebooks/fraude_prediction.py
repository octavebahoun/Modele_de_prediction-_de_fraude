import sys
import os
import pandas as pd

# Ajouter la racine du projet au chemin de recherche Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import load_data, prepare_data

def exploration():
    # Chemins
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, "data/dataset-1.csv")
    
    # 1. Chargement
    df = load_data(data_path)
    
    # 2. Exploration rapide
    print("\n--- Aperçu des données ---")
    print(df.head())
    
    print("\n--- Statistiques descriptives ---")
    print(df.describe())
    
    print("\n--- Répartition des classes (0: Normal, 1: Fraude) ---")
    print(df['is_fraud'].value_counts(normalize=True))

if __name__ == "__main__":
    exploration()