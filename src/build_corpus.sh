#!/bin/bash

# Arrêt du script en cas d'erreur
set -e

# Vérification de l'argument
if [ -z "$1" ]; then
    echo "Usage: ./src/build_corpus.sh <code_langue>"
    echo "Exemple: ./src/build_corpus.sh es"
    exit 1
fi

LANG=$1

# --- CONFIGURATION DES SOURCES (MULTI-MIROIRS) ---
URL_YOUR="https://dumps.wikimedia.your.org/${LANG}wiki/latest/${LANG}wiki-latest-pages-articles.xml.bz2"
URL_UMD="https://mirror.umd.edu/wikimedia/dumps/${LANG}wiki/latest/${LANG}wiki-latest-pages-articles.xml.bz2"
URL_OFF="https://dumps.wikimedia.org/${LANG}wiki/latest/${LANG}wiki-latest-pages-articles.xml.bz2"

DUMP_FILE="donnees/brutes/${LANG}wiki-latest-pages-articles.xml.bz2"
OUT_DIR="donnees/propres/${LANG}"

echo "=========================================================="
echo " TRAITEMENT DE LA LANGUE : $LANG"
echo "=========================================================="

# [1/5] Téléchargement avec aria2c en mode Multi-Sources
echo "[1/5] Téléchargement stabilisé (aria2 multi-miroirs)..."
aria2c -x 6 -s 6 \
       --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
       --continue=true \
       --max-connection-per-server=2 \
       --min-split-size=10M \
       -o "${LANG}wiki-latest-pages-articles.xml.bz2" \
       -d "donnees/brutes/" \
       "$URL_YOUR" "$URL_UMD" "$URL_OFF"

# [2/5] Extraction avec WikiExtractor
echo "[2/5] Extraction et nettoyage des textes (JSON)..."
python3 -m wikiextractor.WikiExtractor $DUMP_FILE -o $OUT_DIR --json

# [3/5] Insertion dans PostgreSQL
echo "[3/5] Insertion des 500 articles dans PostgreSQL..."
python3 src/insertion_db.py $LANG

# [4/5] Nettoyage des textes en base
echo "[4/5] Nettoyage des textes en base de données..."
python3 src/nettoyage.py $LANG

# [5/5] Suppression de l'archive lourde
echo "[5/5] Suppression de l'archive lourde ($DUMP_FILE)..."
rm -f $DUMP_FILE

echo "=========================================================="
echo " SUCCÈS ! Le corpus '$LANG' est prêt et nettoyé."
echo "=========================================================="