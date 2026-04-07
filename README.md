# 🌍 Détection de Langues par Réseau de Neurones (MLP from scratch)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Deep_Learning-777BB4.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Data_Engineering-336791.svg)
![Keras](https://img.shields.io/badge/Keras-Benchmark-D00000.svg)

Ce dépôt contient l'intégralité du code source d'un projet de fin de Licence en Informatique. L'objectif est de concevoir, d'entraîner et d'évaluer un système de détection automatique de la langue d'un texte par apprentissage profond.

La particularité et le défi de ce projet résident dans son approche de bas niveau : **le réseau de neurones (Perceptron Multicouche) a été codé intégralement "from scratch" en utilisant exclusivement les opérations matricielles de la bibliothèque NumPy**, incluant l'initialisation des poids, la propagation avant et la rétropropagation du gradient mathématique exact.

---

## 🎯 Performances et Résultats (Benchmark)

Le modèle a été entraîné à classifier des textes parmi **8 langues cibles** aux alphabets et structures variés (Arabe, Allemand, Anglais, Espagnol, Français, Néerlandais, Portugais, Russe).

| Système évalué | Optimiseur & Paramètres | Accuracy (Jeu de test) |
| :--- | :--- | :---: |
| **Notre MLP NumPy** | SGD (lr=0.1, 200 époques) | **97.67 %** |
| Keras (Base stricte) | SGD (lr=0.1, 200 époques) | 88.33 % |
| Keras (Optimisé) | Adam + BatchNorm (50 époques) | 99.83 % |
| `langdetect` (Pro) | Modèle bayésien pré-entraîné | 99.83 % |

Notre implémentation "fait-main" dépasse la configuration de base de TensorFlow/Keras et s'approche des standards de l'industrie, démontrant la viabilité de l'architecture algorithmique.

---

## ⚙️ Pipeline de Données (Data Engineering)

Le projet intègre une véritable chaîne de traitement de données pour constituer un corpus robuste de **4 000 articles** (1,56 million de mots).

1. **Acquisition stabilisée :** Téléchargement des *dumps* officiels de Wikipédia via `aria2c` (multi-miroirs).
2. **Extraction & Nettoyage :** Utilisation de `WikiExtractor` pour purger le balisage XML/Wiki, complété par des expressions régulières (Regex) en Python pour supprimer les URLs et caractères de contrôle.
3. **Stockage relationnel :** Centralisation dans une base **PostgreSQL**. L'intégrité est garantie par un contrôle d'encodage (100% UTF-8) et la génération de checksums SHA-256 pour éviter tout doublon (zéro *Data Leakage*).
4. **Vectorisation (Bag-of-Words) :** Extraction des signatures linguistiques via la fréquence relative d'apparition des 300 n-grammes de caractères (bigrammes et trigrammes) les plus fréquents.

---

## 🧠 Architecture du Modèle (NumPy)

L'architecture mathématique codée dans `src/mlp_numpy.py` comprend :
* **Couche d'entrée :** Vecteur de dimension 300 (vocabulaire des n-grammes).
* **Couches cachées :** 128 neurones puis 64 neurones (Fonction d'activation **ReLU**).
* **Couche de sortie :** 8 neurones (Fonction d'activation **Softmax** stabilisée).
* **Optimisation :** Descente de gradient stochastique (SGD) par mini-batch (32).
* **Fonction de coût :** Entropie Croisée (*Cross-Entropy*).
* **Initialisation :** Méthode de Xavier/Glorot pour adapter la variance à la taille des couches.

---

## 📂 Structure du Projet

<pre>
projet_langue/
├── notebooks/                       # Environnement d'expérimentation et d'analyse
│   ├── exploration_corpus.ipynb     # Validation SQL, encodage et analyse des n-grammes
│   ├── entrainement_mlp.ipynb       # Grid Search des hyperparamètres (Époques vs LR)
│   ├── comparaison_keras.ipynb      # Benchmark chronométré face à TensorFlow/Keras
│   └── evaluation_finale.ipynb      # Matrice de confusion et analyse des faux positifs
├── src/                             # Code source applicatif
│   ├── build_corpus.sh              # Extraction automatisée WikiExtractor
│   ├── insertion_db.py              # Hashage SHA-256 et insertion PostgreSQL
│   ├── nettoyage.py                 # Purge Regex et filtre de longueur (>200 chars)
│   ├── features.py                  # Construction du vocabulaire et split Train/Val/Test
│   ├── mlp_numpy.py                 # Le réseau de neurones from scratch
│   └── mlp_keras.py                 # La baseline framework pour comparaison
├── requirements.txt                 # Dépendances et librairies Python
├── run_all.sh                       # Script d'orchestration globale de la pipeline
└── README.md
</pre>

*(Note : Les dossiers `donnees/` et `modeles/` sont générés localement par les scripts et ignorés par Git en raison de leur poids).*

---

## 🚀 Utilisation et Reproductibilité

L'ensemble de la démarche est rigoureusement documenté et reproductible. 

**Prérequis :** Python 3.8+, PostgreSQL, `aria2c`.

1. **Cloner le dépôt :**
   <pre>
   git clone https://github.com/heavyrainMK/projet_langue.git
   cd projet_langue
   pip install -r requirements.txt
   </pre>

2. **Reconstituer la base de données (Pipeline complète) :**
   Le script d'orchestration télécharge, extrait, nettoie et insère les données pour les 8 langues cibles automatiquement.
   <pre>
   chmod +x run_all.sh
   ./run_all.sh
   </pre>

3. **Générer les tenseurs et le vocabulaire :**
   <pre>
   python3 src/features.py
   </pre>

4. **Explorer et évaluer :**
   Lancez un serveur Jupyter et exécutez les fichiers présents dans le dossier `notebooks/` de manière séquentielle pour revivre l'optimisation, l'entraînement et l'analyse critique de la matrice de confusion.

---
*Projet réalisé par Maxim Khomenko (2025-2026).*