import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def setup_output_dir(directory):
    """Crée le dossier de sortie s'il n'existe pas."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Dossier créé : {directory}")

def plot_feature_importance(model, feature_names, output_path):
    """Génère un graphique d'importance des variables."""
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    sns.barplot(x=importances, y=feature_names, hue=feature_names, palette='viridis', legend=False)
    plt.title("Impact des caractéristiques sur la détection")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Importance des variables sauvegardée : {output_path}")

def plot_correlation_matrix(df, output_path):
    """Génère une matrice de corrélation."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0)
    plt.title("Corrélations entre variables")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Matrice de corrélation sauvegardée : {output_path}")

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Génère une matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    # Suppression du plot() direct pour mieux contrôler la figure
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraude'])
    disp.plot(cmap='Blues')
    plt.title("Bilan des prédictions (Matrice de Confusion)")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Matrice de confusion sauvegardée : {output_path}")
