#!/bin/bash

# Les 8 langues de notre corpus (Projet de Licence)
LANGUES=("ar" "de" "en" "es" "fr" "nl" "pt" "ru")

for l in "${LANGUES[@]}"
do
    echo "=========================================================="
    echo " Lancement de la pipeline pour : $l"
    echo "=========================================================="
    
    # On lance l'usine à données
    ./src/build_corpus.sh $l
    
    # Pause de 10 secondes pour ne pas spammer et bannir notre IP des serveurs Wikimedia
    sleep 10
done

echo "TOUTES LES LANGUES SONT EN BASE DE DONNÉES !"