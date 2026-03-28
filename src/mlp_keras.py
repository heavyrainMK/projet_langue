import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Masque les logs inutiles de TensorFlow

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input # type: ignore
from tensorflow.keras.optimizers import SGD, Adam # type: ignore

# ==============================================================================
# FONCTIONS DE CRÉATION DES MODÈLES KERAS
# ==============================================================================

def creer_modele_base(n_entree, n_classes, taux_apprentissage=0.1):
    """
    Crée l'équivalent strict du MLP NumPy (Partie 5).
    Architecture : Entrée -> 128 (ReLU) -> 64 (ReLU) -> Sortie (Softmax)
    Optimiseur : SGD classique (Descente de gradient stochastique).
    """
    modele = Sequential([
        Input(shape=(n_entree,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    optimiseur = SGD(learning_rate=taux_apprentissage)
    modele.compile(optimizer=optimiseur, 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])
    
    return modele

def creer_modele_optimise(n_entree, n_classes, taux_apprentissage=0.001):
    """
    Crée une version optimisée du MLP avec les outils modernes de Keras.
    Ajouts : BatchNormalization, Dropout (0.3), et optimiseur Adam.
    """
    modele = Sequential([
        Input(shape=(n_entree,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(n_classes, activation='softmax')
    ])
    
    # Adam gère son propre taux d'apprentissage de manière dynamique
    optimiseur = Adam(learning_rate=taux_apprentissage)
    modele.compile(optimizer=optimiseur, 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])
    
    return modele

# ==============================================================================
# TEST RAPIDE SUR UN PETIT ÉCHANTILLON (Validation technique)
# ==============================================================================

if __name__ == "__main__":
    import numpy as np
    print("=== Test de la mécanique Keras sur un échantillon ===")
    
    try:
        X_train = np.load("donnees/features/X_train.npy")
        y_train = np.load("donnees/features/y_train.npy")
        
        # Petit échantillon pour valider que la mécanique tourne
        X_petit, y_petit = X_train[:50], y_train[:50]
        n_entree = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        
        print("1. Test du modèle de base (équivalent NumPy)...")
        modele_base = creer_modele_base(n_entree, n_classes)
        modele_base.fit(X_petit, y_petit, epochs=2, batch_size=10, verbose=0)
        
        print("2. Test du modèle optimisé (Adam, Dropout)...")
        modele_opti = creer_modele_optimise(n_entree, n_classes)
        modele_opti.fit(X_petit, y_petit, epochs=2, batch_size=10, verbose=0)
        
        print("\nFichier src/mlp_keras.py fonctionnel. Prêt pour la comparaison expérimentale !")
        
    except FileNotFoundError:
        print("Erreur : Fichiers de données introuvables. Lancez depuis la racine du projet.")