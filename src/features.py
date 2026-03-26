import os
import sys
import numpy as np
import psycopg2
import json
from collections import Counter

# Configuration base de données
DB_PARAMS = {
    "dbname": "langdetect_projet",
    "user": "langdetect_user",
    "password": "motdepasse",
    "host": "localhost",
    "port": "5432"
}

# Langues cibles et leur index numérique
LANGUES = {
    "ar": 0, "de": 1, "en": 2, "es": 3,
    "fr": 4, "nl": 5, "pt": 6, "ru": 7
}

VOCAB_SIZE = 300  # Nombre de n-grammes à conserver

def extraire_ngrammes(texte, n_valeurs=[2, 3]):
    """
    Extrait tous les bigrammes et trigrammes de caractères d'un texte.
    Retourne une liste de n-grammes.
    """
    texte = texte.lower()
    ngrammes = []
    for n in n_valeurs:
        ngrammes += [texte[i:i+n] for i in range(len(texte) - n + 1)]
    return ngrammes

def construire_vocabulaire(textes, vocab_size=VOCAB_SIZE):
    """
    Construit le vocabulaire global : les vocab_size n-grammes
    les plus fréquents sur l'ensemble des textes.
    """
    print(f"Construction du vocabulaire ({vocab_size} n-grammes)...")
    compteur = Counter()
    for texte in textes:
        compteur.update(extraire_ngrammes(texte))
    
    vocabulaire = [ng for ng, _ in compteur.most_common(vocab_size)]
    print(f"Vocabulaire construit : {len(vocabulaire)} n-grammes.")
    return vocabulaire

def vectoriser(texte, vocabulaire):
    """
    Transforme un texte en vecteur de fréquences relatives.
    Retourne un vecteur numpy de dimension len(vocabulaire).
    """
    ngrammes = extraire_ngrammes(texte)
    total = len(ngrammes)
    
    if total == 0:
        return np.zeros(len(vocabulaire))
    
    compteur = Counter(ngrammes)
    vecteur = np.array([
        compteur.get(ng, 0) / total
        for ng in vocabulaire
    ], dtype=np.float32)
    
    return vecteur

def charger_donnees():
    """
    Charge tous les textes et labels depuis PostgreSQL.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_PARAMS["dbname"],
            user=DB_PARAMS["user"],
            password=DB_PARAMS["password"],
            host=DB_PARAMS["host"],
            port=DB_PARAMS["port"]
        )
        cur = conn.cursor()
        cur.execute("SELECT contenu, langue FROM documents ORDER BY langue")
        rows = cur.fetchall()
        
        textes = [row[0] for row in rows]
        labels = [row[1] for row in rows]
        
        print(f"Données chargées : {len(textes)} articles.")
        
        # Afficher la répartition
        compteur = Counter(labels)
        for langue, nb in sorted(compteur.items()):
            print(f"  {langue} : {nb} articles")
        
        return textes, labels
    
    finally:
        if conn:
            conn.close()

def labels_vers_indices(labels):
    """
    Convertit les codes de langue en indices numériques.
    Utilise uniquement les langues présentes dans les données.
    """
    langues_presentes = sorted(set(labels))
    mapping = {langue: i for i, langue in enumerate(langues_presentes)}
    print(f"\nMapping langues → indices : {mapping}")
    return np.array([mapping[l] for l in labels], dtype=np.int32), mapping

def diviser_dataset(X, y, ratio_train=0.70, ratio_val=0.15):
    """
    Divise le dataset en train / validation / test.
    70% / 15% / 15%
    """
    n = len(X)
    indices = np.random.permutation(n)
    
    n_train = int(n * ratio_train)
    n_val = int(n * ratio_val)
    
    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]
    
    return (
        X[idx_train], X[idx_val], X[idx_test],
        y[idx_train], y[idx_val], y[idx_test]
    )

def main():
    print("=" * 55)
    print("   EXTRACTION DES FEATURES")
    print("=" * 55)
    
    # 1. Charger les données
    textes, labels = charger_donnees()
    
    # 2. Construire le vocabulaire
    vocabulaire = construire_vocabulaire(textes, VOCAB_SIZE)
    
    # 3. Vectoriser tous les textes
    print(f"\nVectorisation de {len(textes)} textes...")
    X = np.array([vectoriser(t, vocabulaire) for t in textes], dtype=np.float32)
    print(f"Matrice X : {X.shape}")
    
    # 4. Convertir les labels
    y, mapping = labels_vers_indices(labels)
    
    # 5. Diviser en train / val / test
    np.random.seed(42)  # Pour la reproductibilité
    X_train, X_val, X_test, y_train, y_val, y_test = diviser_dataset(X, y)
    
    print(f"\nDivision du dataset :")
    print(f"  Train      : {X_train.shape[0]} articles ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"  Validation : {X_val.shape[0]} articles ({X_val.shape[0]/len(X)*100:.0f}%)")
    print(f"  Test       : {X_test.shape[0]} articles ({X_test.shape[0]/len(X)*100:.0f}%)")
    
    # 6. Sauvegarder les fichiers .npy
    dossier = "donnees/features"
    os.makedirs(dossier, exist_ok=True)
    
    np.save(f"{dossier}/X_train.npy", X_train)
    np.save(f"{dossier}/X_val.npy",   X_val)
    np.save(f"{dossier}/X_test.npy",  X_test)
    np.save(f"{dossier}/y_train.npy", y_train)
    np.save(f"{dossier}/y_val.npy",   y_val)
    np.save(f"{dossier}/y_test.npy",  y_test)
    
    # 7. Sauvegarder le vocabulaire et le mapping
    with open(f"{dossier}/vocabulaire.json", 'w', encoding='utf-8') as f:
        json.dump(vocabulaire, f, ensure_ascii=False)
    
    with open(f"{dossier}/mapping_langues.json", 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False)
    
    print(f"\nFichiers sauvegardés dans '{dossier}/' :")
    for fichier in os.listdir(dossier):
        taille = os.path.getsize(f"{dossier}/{fichier}")
        print(f"  {fichier:<25} {taille/1024:.1f} KB")
    
    print("\nPartie 4 terminée !")

if __name__ == "__main__":
    main()