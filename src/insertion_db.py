import os
import sys
import json
import psycopg2
import hashlib
from datetime import datetime
import glob

# Configuration de la base de données
DB_PARAMS = {
    "dbname": "langdetect_projet",
    "user": "langdetect_user",
    "password": "motdepasse",
    "host": "localhost",
    "port": "5432"
}

def creer_table_si_besoin(cur):
    """Crée la table documents selon le schéma de votre plan."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            langue VARCHAR(10),
            contenu TEXT,
            longueur INT,
            source VARCHAR(100),
            date_recolte TIMESTAMP,
            checksum VARCHAR(64)
        );
    """)

def generer_checksum(texte):
    """Génère une empreinte unique pour éviter les doublons."""
    return hashlib.sha256(texte.encode('utf-8')).hexdigest()

def inserer_corpus(langue, dossier_source, limite=500):
    print(f"Connexion à PostgreSQL pour insérer la langue : {langue}...")
    
    # Initialisation explicite pour satisfaire l'analyseur de code (Pylance)
    conn = None
    cur = None
    
    try:
        # Déroulement explicite des paramètres de connexion
        conn = psycopg2.connect(
            dbname=DB_PARAMS["dbname"],
            user=DB_PARAMS["user"],
            password=DB_PARAMS["password"],
            host=DB_PARAMS["host"],
            port=DB_PARAMS["port"]
        )
        cur = conn.cursor()
        
        creer_table_si_besoin(cur)
        conn.commit()
        
        # On cherche tous les fichiers générés par Wikiextractor (ex: AA/wiki_00)
        chemin_fichiers = os.path.join(dossier_source, "**", "wiki_*")
        fichiers = glob.glob(chemin_fichiers, recursive=True)
        
        if not fichiers:
            print(f"Aucun fichier trouvé dans {dossier_source}")
            return

        textes_inseres = 0
        
        for fichier in fichiers:
            if textes_inseres >= limite:
                break
                
            with open(fichier, 'r', encoding='utf-8') as f:
                for ligne in f:
                    if textes_inseres >= limite:
                        break
                        
                    try:
                        # Wikiextractor génère un objet JSON par ligne
                        article = json.loads(ligne)
                        texte = article.get("text", "").strip()
                        
                        # On filtre les textes trop courts selon votre plan (> 200)
                        longueur = len(texte)
                        if longueur > 200:
                            source = article.get("url", f"Wikipedia_{langue}")
                            checksum = generer_checksum(texte)
                            date_recolte = datetime.now()
                            
                            # Insertion sécurisée avec des paramètres pour éviter les injections SQL
                            cur.execute("""
                                INSERT INTO documents (langue, contenu, longueur, source, date_recolte, checksum)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """, (langue, texte, longueur, source, date_recolte, checksum))
                            
                            textes_inseres += 1
                            
                            if textes_inseres % 100 == 0:
                                print(f"Progression : {textes_inseres}/{limite} articles insérés.")
                                conn.commit() # On sauvegarde par lots de 100
                                
                    except json.JSONDecodeError:
                        continue # On ignore les lignes mal formatées s'il y en a
        
        # Sauvegarde finale
        conn.commit()
        print(f"\nTerminé ! {textes_inseres} articles en '{langue}' ont été insérés dans PostgreSQL.")
        
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
    finally:
        # Fermeture propre et sécurisée des connexions
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    # Vérification qu'une langue a bien été passée en argument dans le terminal
    if len(sys.argv) != 2:
        print("Usage: python3 src/insertion_db.py <code_langue>")
        sys.exit(1)
        
    langue_cible = sys.argv[1]
    dossier_cible = f"donnees/propres/{langue_cible}"
    
    inserer_corpus(langue_cible, dossier_cible, limite=500)