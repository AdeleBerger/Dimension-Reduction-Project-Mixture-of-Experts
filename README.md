# Sentiment Analysis via MoE and Representation of IMDb Movie Reviews

## Introduction

L'analyse de sentiment est une tâche fondamentale en traitement du langage naturel (NLP), visant à déterminer le sentiment exprimé dans un texte. Ce projet explore la classification des critiques de films en utilisant le jeu de données IMDb et implémente un modèle **Mixture of Experts (MoE)** pour améliorer la performance de classification.

Le projet utilise le jeu de données des critiques de films IMDb, disponible sur plusieurs plateformes, dont Hugging Face et Stanford AI. L'objectif est de classer les critiques de films comme étant positives ou négatives.

## Dataset

Le jeu de données utilisé dans ce projet est le **IMDb Movie Reviews Dataset**, qui contient 50 000 critiques de films étiquetées comme positives ou négatives. Les critiques sont réparties de manière égale entre un jeu d'entraînement et un jeu de test.

- **Jeu de données original** : [Stanford AI - IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Jeu de données avec références** : [Hugging Face - IMDb Dataset](https://huggingface.co/datasets/imdb)
- **Article de référence** : [Learning Word Vectors for Sentiment Analysis (Maas et al., 2011)](https://ai.stanford.edu/~amaas/papers/amaas-icml-2011.pdf)

## Objectifs du projet

1. **Préparation des données**
   - Télécharger et prétraiter le jeu de données.
   - Sélectionner un échantillon aléatoire de 2 000 critiques (1 000 positives et 1 000 négatives) pour un traitement plus rapide.
   - Prétraitement du texte : tokenisation, suppression des stopwords, stemming/lemmatisation.

2. **Extraction des caractéristiques**
   - Convertir les données textuelles en représentations numériques :
     - Vectorisation **TF-IDF**.
     - Utilisation des **Word embeddings** (Word2Vec, GloVe).

3. **Visualisation**
   - Utiliser **t-SNE** et **UMAP** pour représenter les données et mieux comprendre les relations entre les critiques.

4. **Implémentation du modèle : Mixture of Experts (MoE)**
   - Entraîner un modèle MoE avec des hyperparamètres optimisés.
   - Comparer la performance avec les modèles de base :
     - **Régression logistique**.
     - **Réseaux de neurones** (MLP, CNN, LSTM).
   - Évaluer les modèles en utilisant des métriques telles que **précision**, **rappel**, **F1-score**.

5. **Analyse**
   - Analyser l'affectation des experts pour différents types de critiques.
   - Visualiser les frontières de décision et le mécanisme de routage.

6. **Résultats et discussion**
   - Comparaison des performances entre MoE et les modèles traditionnels.
   - Interprétabilité : Comment les différents experts contribuent à la classification.
   - Suggestions d'améliorations potentielles.

## Installation et Prérequis

Pour exécuter ce projet, vous devez disposer de l'environnement Python suivant et des bibliothèques nécessaires :

### Prérequis :
- Python 3.x
- `pandas` - pour la gestion des données
- `numpy` - pour les calculs numériques
- `scikit-learn` - pour les algorithmes d'apprentissage automatique
- `tensorflow` / `pytorch` - pour l'entraînement des réseaux de neurones
- `gensim` - pour les Word embeddings (Word2Vec)
- `nltk` - pour le prétraitement du texte
- `matplotlib` / `seaborn` - pour la visualisation des résultats
- `umap-learn` et `sklearn.manifold` - pour les réductions de dimension (t-SNE et UMAP)

### Installation des dépendances :
```bash
pip install pandas numpy scikit-learn tensorflow gensim nltk matplotlib seaborn umap-learn
```

