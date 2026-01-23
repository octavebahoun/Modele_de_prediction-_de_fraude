# Projet de Détection de Fraude

Ce projet vise à identifier les transactions frauduleuses en utilisant des techniques d'apprentissage automatique.

## Structure du Projet

- **data/** : Contient les jeux de données (CSV).
- **notebooks/** : Code python pour l'exploration des données et l'entraînement des modèles.
- **models/** : Modèles entraînés sauvegardés (format .joblib).
- **outputs/** : Visualisations et graphiques générés (matrices de confusion, importance des variables).
- **presentation/** : Support de présentation final.
- **requirements.txt** : Liste des dépendances Python.

## Installation

```bash
# Créer l'environnement virtuel (déjà fait)
python3 -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate

# Installer les dépendances (déjà fait)
pip install -r requirements.txt
```

## Utilisation

Pour lancer le script avec l'environnement virtuel :

```bash
./venv/bin/python notebooks/fraude_prediction.py
```
