#!/bin/bash
# Démarrer ollama en arrière-plan
ollama serve & 
# pid=$!

echo "Attente du démarrage du serveur Ollama..."
sleep 5  # Attendre 5 secondes pour s'assurer qu'Ollama est prêt

echo "Pulling model gemma3:1b..." # on pull gemma
ollama pull gemma3:1b

#  nomic-embed-text
echo "Pulling model nomic-embed-text..."
ollama pull nomic-embed-text

# Attendre le processus
# wait $pid

# kill $pid  # Important pour que le build se termine et sauvegarde !


