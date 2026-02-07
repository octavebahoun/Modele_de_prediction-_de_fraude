# Modèle de Détection de Fraude — Synthèse pour réunion

## Objectif métier

Détecter automatiquement les transactions suspectes afin de **réduire les pertes financières** et **prioriser les contrôles** humains.

## Ce que fait le modèle

- **Prédit** si une transaction est frauduleuse ou non.
- **Classe** les transactions par niveau de risque.
- **Explique** les facteurs les plus influents (importance des variables).

## Données utilisées

- Jeu de données tabulaire (CSV) stocké dans [data/](data/).
- Variables de type montant, fréquence, historique, comportement, etc.
- Nettoyage et préparation effectués dans les scripts de [notebooks/](notebooks/).

## Approche technique (résumé)

- Modèle principal : **Random Forest**.
- Entraînement et sauvegarde dans [models/](models/).
- Évaluation via **matrice de confusion** et **importance des variables** (sorties dans [outputs/](outputs/)).

## Résultats clés (à présenter)

- **Performance globale** : précision, rappel, F1.
- **Priorisation des alertes** : top transactions à vérifier.
- **Variables les plus discriminantes** : top 5–10.

## Démo rapide (optionnel)

Pour exécuter la prédiction sur un exemple local :

```bash
./venv/bin/python notebooks/fraude_prediction.py
```

## Structure du projet

- [data/](data/) : jeux de données (CSV).
- [notebooks/](notebooks/) : scripts d’exploration et d’entraînement.
- [models/](models/) : modèles entraînés (.joblib).
- [outputs/](outputs/) : graphiques et résultats.
- [presentation/](presentation/) : support final.
- [requirements.txt](requirements.txt) : dépendances Python.

## Prochaines étapes proposées

- Ajuster le **seuil d’alerte** selon la capacité opérationnelle.
- Ajouter un **monitoring** de performance en production.
- Mettre en place un **retraining** périodique.
