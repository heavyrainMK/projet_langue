#!/bin/bash

# Les 6 langues restantes de votre plan
LANGUES=("en" "de" "nl" "ru" "ar")

for l in "${LANGUES[@]}"
do
    echo "=========================================================="
    echo " Lancement de la pipeline pour : $l"
    echo "=========================================================="
    
    # On lance l'usine à données
    ./src/build_corpus.sh $l
    
    # Pause de 10 secondes pour ne pas spammer les serveurs
    sleep 10
done

echo "TOUTES LES LANGUES SONT EN BASE DE DONNÉES !"