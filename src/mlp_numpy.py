import numpy as np

# ==============================================================================
# FONCTIONS D'ACTIVATION ET LEURS DÉRIVÉES
# ==============================================================================

def relu(z):
    """
    Fonction d'activation ReLU.
    Retourne max(0, z) élément par élément.
    """
    return np.maximum(0, z)

def relu_derivee(z):
    """
    Dérivée de ReLU.
    Retourne 1 si z > 0, 0 sinon.
    """
    return (z > 0).astype(np.float32)

def softmax(z):
    """
    Fonction d'activation Softmax.
    Convertit un vecteur de scores en probabilités.
    La soustraction de max(z) évite les débordements numériques.
    """
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ==============================================================================
# FONCTION DE COÛT
# ==============================================================================

def entropie_croisee(y_pred, y_vrai):
    """
    Calcule l'entropie croisée entre les prédictions et les vraies étiquettes.
    y_pred : matrice de probabilités (n_exemples, n_classes)
    y_vrai : vecteur d'indices de classes (n_exemples,)
    """
    n = y_pred.shape[0]
    # On clip pour éviter log(0)
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # On récupère la probabilité de la vraie classe pour chaque exemple
    log_probs = np.log(y_pred_clip[np.arange(n), y_vrai])
    return -np.mean(log_probs)

# ==============================================================================
# CLASSE MLP
# ==============================================================================

class MLP:
    """
    Perceptron Multicouche (Multi-Layer Perceptron) implémenté avec NumPy.
    
    Architecture :
        Entrée (n_entree) 
        → Couche cachée 1 (128 neurones, ReLU)
        → Couche cachée 2 (64 neurones, ReLU)
        → Sortie (n_classes, Softmax)
    """
    
    def __init__(self, n_entree, n_classes, n_cache1=128, n_cache2=64, taux_apprentissage=0.1):
        """
        Initialise les poids et biais du réseau.
        
        Paramètres :
            n_entree           : dimension du vecteur d'entrée (300 pour nous)
            n_classes          : nombre de langues à distinguer
            n_cache1           : neurones dans la première couche cachée
            n_cache2           : neurones dans la deuxième couche cachée
            taux_apprentissage : learning rate pour la descente de gradient
        """
        self.taux = taux_apprentissage
        self.n_classes = n_classes
        
        # Initialisation de Xavier/Glorot pour éviter l'explosion/disparition du gradient
        # Les poids sont initialisés avec une variance adaptée à la taille des couches
        self.W1 = np.random.randn(n_entree, n_cache1) * np.sqrt(2.0 / n_entree)
        self.b1 = np.zeros((1, n_cache1))
        
        self.W2 = np.random.randn(n_cache1, n_cache2) * np.sqrt(2.0 / n_cache1)
        self.b2 = np.zeros((1, n_cache2))
        
        self.W3 = np.random.randn(n_cache2, n_classes) * np.sqrt(2.0 / n_cache2)
        self.b3 = np.zeros((1, n_classes))
        
        # Historique pour les courbes d'apprentissage
        self.historique = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [],  'val_acc': []
        }
    
    # --------------------------------------------------------------------------
    # PROPAGATION AVANT
    # --------------------------------------------------------------------------
    
    def forward(self, X):
        """
        Propagation avant : calcule la sortie du réseau pour une entrée X.
        Sauvegarde les valeurs intermédiaires pour la rétropropagation.
        """
        # Couche cachée 1
        self.Z1 = X @ self.W1 + self.b1      # Combinaison linéaire
        self.A1 = relu(self.Z1)               # Activation ReLU
        
        # Couche cachée 2
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        
        # Couche de sortie
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = softmax(self.Z3)            # Activation Softmax → probabilités
        
        return self.A3
    
    # --------------------------------------------------------------------------
    # RÉTROPROPAGATION
    # --------------------------------------------------------------------------
    
    def backward(self, X, y):
        """
        Rétropropagation du gradient.
        Calcule les gradients de la fonction de coût par rapport à tous les poids.
        
        Astuce clé (cf. point de vigilance du plan) :
        La dérivée de Softmax + Entropie croisée se simplifie en : ŷ - y
        Ce qui évite des calculs matriciels complexes.
        """
        n = X.shape[0]
        
        # Encodage one-hot des étiquettes
        Y_onehot = np.zeros((n, self.n_classes))
        Y_onehot[np.arange(n), y] = 1
        
        # --- Couche de sortie (gradient simplifié Softmax + Cross-Entropy) ---
        dZ3 = (self.A3 - Y_onehot) / n        # ŷ - y  (simplification clé !)
        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        # --- Couche cachée 2 ---
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivee(self.Z2)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # --- Couche cachée 1 ---
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivee(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # --- Mise à jour des poids (descente de gradient) ---
        self.W3 -= self.taux * dW3
        self.b3 -= self.taux * db3
        self.W2 -= self.taux * dW2
        self.b2 -= self.taux * db2
        self.W1 -= self.taux * dW1
        self.b1 -= self.taux * db1
    
    # --------------------------------------------------------------------------
    # MÉTHODES UTILITAIRES
    # --------------------------------------------------------------------------
    
    def calculer_accuracy(self, X, y):
        """Calcule la précision du modèle sur un jeu de données."""
        y_pred = np.argmax(self.forward(X), axis=1)
        return np.mean(y_pred == y)
    
    def predict(self, X):
        """
        Prédit la classe pour chaque exemple de X.
        Retourne un vecteur d'indices de classes.
        """
        return np.argmax(self.forward(X), axis=1)
    
    # --------------------------------------------------------------------------
    # ENTRAÎNEMENT
    # --------------------------------------------------------------------------
    
    def fit(self, X_train, y_train, X_val, y_val, n_epoques=100, taille_batch=32, verbose=True):
        """
        Entraîne le réseau par mini-batch gradient descent.
        
        Paramètres :
            X_train, y_train : données d'entraînement
            X_val, y_val     : données de validation
            n_epoques        : nombre de passages sur les données
            taille_batch     : taille des mini-batchs
            verbose          : afficher la progression
        """
        n = X_train.shape[0]
        
        for epoque in range(n_epoques):
            # Décroissance du taux d'apprentissage toutes les 50 époques
            if epoque > 0 and epoque % 50 == 0:
                self.taux *= 0.5
                if verbose:
                    print(f"  → Taux d'apprentissage réduit à {self.taux:.4f}")
            # Mélanger les données à chaque époque
            indices = np.random.permutation(n)
            X_melange = X_train[indices]
            y_melange = y_train[indices]
            
            # Mini-batch gradient descent
            for debut in range(0, n, taille_batch):
                fin = debut + taille_batch
                X_batch = X_melange[debut:fin]
                y_batch = y_melange[debut:fin]
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            
            # Calcul des métriques à la fin de chaque époque
            train_loss = entropie_croisee(self.forward(X_train), y_train)
            val_loss   = entropie_croisee(self.forward(X_val),   y_val)
            train_acc  = self.calculer_accuracy(X_train, y_train)
            val_acc    = self.calculer_accuracy(X_val,   y_val)
            
            self.historique['train_loss'].append(train_loss)
            self.historique['val_loss'].append(val_loss)
            self.historique['train_acc'].append(train_acc)
            self.historique['val_acc'].append(val_acc)
            
            if verbose and (epoque + 1) % 10 == 0:
                print(f"Époque {epoque+1:>4}/{n_epoques} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")
        
        return self
    
    def sauvegarder(self, chemin):
        """Sauvegarde les poids du modèle dans un fichier .npz"""
        np.savez(chemin,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        print(f"Modèle sauvegardé : {chemin}.npz")
    
    def charger(self, chemin):
        """Charge les poids du modèle depuis un fichier .npz"""
        data = np.load(chemin)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
        self.W3, self.b3 = data['W3'], data['b3']
        print(f"Modèle chargé : {chemin}")


# ==============================================================================
# TEST RAPIDE
# ==============================================================================

if __name__ == "__main__":
    print("Chargement des données...")
    
    X_train = np.load("donnees/features/X_train.npy")
    X_val   = np.load("donnees/features/X_val.npy")
    X_test  = np.load("donnees/features/X_test.npy")
    y_train = np.load("donnees/features/y_train.npy")
    y_val   = np.load("donnees/features/y_val.npy")
    y_test  = np.load("donnees/features/y_test.npy")
    
    print(f"X_train : {X_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"X_test  : {X_test.shape}")
    
    n_entree  = X_train.shape[1]   # 300
    n_classes = len(np.unique(y_train))
    
    print(f"\nArchitecture : {n_entree} → 128 → 64 → {n_classes}")
    print(f"Nombre de classes : {n_classes}")
    
    np.random.seed(42)  # Reproductibilité

    # Créer et entraîner le modèle
    modele = MLP(
        n_entree=n_entree,
        n_classes=n_classes,
        taux_apprentissage=0.1
    )
    
    print("\nEntraînement (200 époques)...")
    modele.fit(X_train, y_train, X_val, y_val, n_epoques=200, taille_batch=32)
    
    # Évaluation finale sur le jeu de test
    test_acc = modele.calculer_accuracy(X_test, y_test)
    print(f"\nPrécision sur le jeu de test : {test_acc*100:.2f}%")
    
    # Sauvegarde du modèle
    modele.sauvegarder("modeles/mlp_numpy")
    
    print("\nPartie 5 terminée !")