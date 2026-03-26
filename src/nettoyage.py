import re
import psycopg2

DB_PARAMS = {
    "dbname": "langdetect_projet",
    "user": "langdetect_user",
    "password": "motdepasse",
    "host": "localhost",
    "port": "5432"
}

def nettoyer_texte(texte):
    """
    Applique une série de nettoyages sur un texte brut.
    Retourne le texte nettoyé.
    """
    # 1. Supprimer les URLs résiduelles
    texte = re.sub(r'http\S+|www\.\S+', '', texte)
    
    # 2. Supprimer les caractères de contrôle (sauf retours à la ligne)
    texte = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', texte)
    
    # 3. Remplacer les séquences de whitespace multiples par un seul espace
    texte = re.sub(r'[ \t]+', ' ', texte)
    
    # 4. Remplacer les sauts de ligne multiples par un seul
    texte = re.sub(r'\n{3,}', '\n\n', texte)
    
    # 5. Supprimer les lignes qui ne contiennent que des chiffres ou ponctuation
    lignes = texte.split('\n')
    lignes_propres = [
        ligne for ligne in lignes
        if not re.match(r'^[\d\s\W]+$', ligne) or len(ligne.strip()) == 0
    ]
    texte = '\n'.join(lignes_propres)
    
    # 6. Supprimer les espaces en début et fin
    texte = texte.strip()
    
    return texte

def nettoyer_corpus(langue=None):
    """
    Nettoie tous les textes en base pour une langue donnée.
    Si aucune langue n'est précisée, nettoie tout le corpus.
    """
    conn = None
    cur = None
    
    try:
        conn = psycopg2.connect(
            dbname=DB_PARAMS["dbname"],
            user=DB_PARAMS["user"],
            password=DB_PARAMS["password"],
            host=DB_PARAMS["host"],
            port=DB_PARAMS["port"]
        )
        cur = conn.cursor()
        
        # Récupérer les articles à nettoyer
        if langue:
            cur.execute(
                "SELECT id, contenu FROM documents WHERE langue = %s",
                (langue,)
            )
            print(f"Nettoyage des articles en '{langue}'...")
        else:
            cur.execute("SELECT id, contenu FROM documents")
            print("Nettoyage de tout le corpus...")
        
        articles = cur.fetchall()
        total = len(articles)
        print(f"{total} articles à traiter.")
        
        modifies = 0
        supprimes = 0
        
        for i, (article_id, contenu) in enumerate(articles):
            texte_nettoye = nettoyer_texte(contenu)
            nouvelle_longueur = len(texte_nettoye)
            
            # Si le texte devient trop court après nettoyage, on le supprime
            if nouvelle_longueur < 200:
                cur.execute(
                    "DELETE FROM documents WHERE id = %s",
                    (article_id,)
                )
                supprimes += 1
            else:
                # Sinon on met à jour le contenu et la longueur
                cur.execute(
                    """UPDATE documents 
                       SET contenu = %s, longueur = %s 
                       WHERE id = %s""",
                    (texte_nettoye, nouvelle_longueur, article_id)
                )
                modifies += 1
            
            # Sauvegarde par lots de 100
            if (i + 1) % 100 == 0:
                conn.commit()
                print(f"Progression : {i + 1}/{total} articles traités...")
        
        conn.commit()
        print(f"\nTerminé !")
        print(f"  Articles modifiés  : {modifies}")
        print(f"  Articles supprimés : {supprimes} (trop courts après nettoyage)")
        print(f"  Articles restants  : {modifies}")
        
    except Exception as e:
        print(f"Erreur : {e}")
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        nettoyer_corpus(langue=sys.argv[1])
    else:
        nettoyer_corpus()