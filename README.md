# 🌿 Détecteur de maladies sur feuilles de plantes

> Prends une photo d'une feuille de tomate, poivron ou pomme de terre, et le modèle te dit
> si elle est saine ou identifie la maladie la plus probable — en quelques secondes.

<!-- 
  📸 REMPLACE CETTE LIGNE par une capture d'écran ou un gif de l'app Gradio en action.
  C'est la première chose que les gens verront : un visuel vaut mieux qu'un long texte.
  Exemple : ![Démo de l'application](assets/demo.gif)
-->
![Démo de l'application](assets/demo.gif)

[![Démo live](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face%20Spaces-yellow)](LIEN_A_COMPLETER)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 En bref

Ce projet est un classifieur d'images qui reconnaît **15 maladies** (+ état sain) sur des feuilles
de **tomate, poivron et pomme de terre**, à partir d'une simple photo. Le modèle utilise le
transfer learning (ResNet18 pré-entraîné) et atteint **99% d'accuracy** sur le jeu de test —
un résultat à interpréter avec prudence (voir la section [Limites](#-limites-et-esprit-critique)).

Une interface web (Gradio) permet de tester le modèle directement, sans écrire une ligne de code.

**👉 [Essayer la démo en ligne](LIEN_A_COMPLETER)**

---

## 📋 Sommaire

- [Le problème](#-le-problème)
- [Résultats](#-résultats)
- [Limites et esprit critique](#-limites-et-esprit-critique)
- [Comment ça marche](#-comment-ça-marche)
- [Installation et utilisation](#-installation-et-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Dataset](#-dataset)
- [Pistes d'amélioration](#-pistes-damélioration)
- [Stack technique](#-stack-technique)

---

## 🌱 Le problème

Détecter une maladie sur une plante tôt permet d'agir avant qu'elle ne se propage. Dans un
contexte agricole, ce genre d'outil peut aider à un premier diagnostic rapide, en attendant
l'avis d'un professionnel. C'est aussi un bon cas d'usage pour explorer un pipeline complet
de computer vision : de la donnée brute jusqu'à une application utilisable.

## 📊 Résultats

Évalués sur un jeu de test de 3096 images, jamais vues pendant l'entraînement :

| Métrique | Score |
|---|---|
| Accuracy globale | 99% |
| F1-score (macro avg) | 0.99 |
| F1-score (classe la plus rare, *Potato — Sain*, 152 images) | 0.98 |

 ![Matrice de confusion](assets/confusion_matrix.png) 

La gestion du déséquilibre des classes (jusqu'à 21x d'écart entre la classe la plus et la
moins représentée) via une loss pondérée a permis de maintenir de bonnes performances même
sur les classes minoritaires.

## ⚠️ Limites et esprit critique

Un score de 99% est excellent, mais aussi un signal à interroger plutôt qu'à célébrer
aveuglément. Le dataset utilisé ([PlantVillage](#-dataset)) est constitué de photos prises
**en conditions de laboratoire** : fond uniforme, feuille isolée, éclairage contrôlé.

C'est une limite connue et documentée dans la littérature scientifique : des modèles entraînés
sur PlantVillage avec des scores proches de 99% voient souvent leurs performances chuter
significativement sur des photos de terrain réelles (fond complexe, éclairage variable, feuille
partiellement visible).

**Conclusion honnête** : ce projet démontre une maîtrise du pipeline de bout en bout
(exploration → entraînement → déploiement), mais n'est pas prêt pour un usage terrain sans
ré-entraînement sur des données plus réalistes.

## ⚙️ Comment ça marche

1. **Exploration des données** (`notebooks/01_exploration.ipynb`) — analyse de la distribution
   des classes, détection d'images corrompues, aperçu visuel.
2. **Entraînement** (`notebooks/02_training.ipynb`) — transfer learning sur un ResNet18
   pré-entraîné (ImageNet), avec loss pondérée pour compenser le déséquilibre des classes.
3. **Déploiement** (`app/app.py`) — interface Gradio qui charge le modèle et renvoie les
   3 hypothèses les plus probables avec leur score de confiance.

## 🚀 Installation et utilisation

```bash
# Cloner le repo
git clone https://github.com/TON_USERNAME/plant-disease-classifier.git
cd plant-disease-classifier

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'interface
cd app
python app.py
```

L'interface s'ouvre sur `http://127.0.0.1:7860`.

Pour ré-entraîner le modèle toi-même, télécharge le [dataset PlantVillage](#-dataset), place-le
à la racine du projet, puis exécute les notebooks dans l'ordre (`01_exploration.ipynb` puis
`02_training.ipynb`).

## 📁 Structure du projet

```
plant-disease-classifier/
├── README.md
├── requirements.txt
├── 01_exploration.ipynb    # Analyse du dataset
├── 02_training.ipynb       # Entraînement du modèle
├── app.py                  # Interface Gradio
├── models/
│   └── best_model.pth          # Poids du modèle entraîné
└── assets/
    └── ...                     # Captures d'écran, gifs pour ce README
```

## 🗂️ Dataset

[PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) — environ 20 000 images
réparties sur 15 classes (maladies + état sain) pour la tomate, le poivron et la pomme de terre.

## 🔭 Pistes d'amélioration

- **Robustesse terrain** : ré-entraîner ou fine-tuner sur des photos prises en conditions
  réelles (fond complexe, extérieur) pour valider la limite identifiée ci-dessus.
- **Grad-CAM / explicabilité** : visualiser sur quelles zones de l'image le modèle se base
  pour décider — vérifierait s'il regarde bien les symptômes et pas le fond.
- **Élargir les espèces** : le dataset PlantVillage couvre d'autres cultures (maïs, raisin,
  pomme...) qui pourraient être intégrées.
- **Quantization / optimisation** : rendre le modèle plus léger pour un déploiement mobile.

## 🛠️ Stack technique

- **Modèle** : PyTorch, ResNet18 (via `timm`)
- **Traitement de données** : torchvision, Pillow
- **Évaluation** : scikit-learn
- **Interface** : Gradio
- **Visualisation** : Matplotlib, Seaborn

---


*N'hésite pas à me contacter pour toute question : [vergne.clement49@gmail.com]*
