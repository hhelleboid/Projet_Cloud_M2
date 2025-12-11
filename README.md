# Projet_Cloud_M2

# 🤖 Assistant Documentaire RAG (Local)

Ce projet est un assistant intelligent (RAG - Retrieval Augmented Generation) capable de répondre à des questions basées sur vos propres documents PDF.

Il fonctionne **entièrement en local** en utilisant **Ollama** pour le LLM et les embeddings et **ChromaDB** pour la base de données vectorielle.

## ✨ Fonctionnalités

- **Upload de PDF** : Ajoutez vos documents directement via l'interface.
- **Indexation automatique** : Découpage intelligent (Chunking) et vectorisation des textes.
- **Recherche Hybride** : Utilise la recherche vectorielle + un Reranking pour une précision maximale.
- **Historique de chat** : Sauvegarde automatique de la conversation.
- **100% Local** : Aucune donnée n'est envoyée dans le cloud (nécessite un bon CPU).

---

## 🛠️ Prérequis

Avant de commencer, assurez-vous d'avoir installé :

1. **Python 3.10+** : [Télécharger Python](https://www.python.org/downloads/)
2. **Ollama** : [Télécharger Ollama](https://ollama.com/)
3. **Git** (pour cloner le projet).

---

## 🚀 Installation

### 1. Cloner ou télécharger le projet

```bash
git clone <votre-repo-url>
cd Projet_Cloud_M2/app
```

### 2. Créer l'environnement virtuel

Sous window : 

```bash
python -m venv venv .\venv\Scripts\activate
```

Sous Mac/Linux :

```bash
python3 -m venv venv source venv/bin/activate
```

### 3. Installer les dépendances

Pour installer les dépendances effectuer la commande suivante :

```bash
pip install -r requirements.txt
```

## 🦙 Configuration des Modèles Ollama

Ce projet nécessite deux types de modèles pour fonctionner. Vous devez les télécharger via votre terminal (CMD ou PowerShell) une fois Ollama installé.

### 1. Modèle d'Embedding 

Sert à transformer le texte en vecteurs mathématiques.

```bash
ollama pull nomic-embed-text
```

### 2. Modèle de Langage (LLM)

```bash 
ollama pull gemma3:1b
```

Note : Assurez-vous que l'application Ollama tourne en arrière-plan (icône dans la barre des tâches) ou lancez ``ollama serve`` dans un terminal séparé.


## ⚙️ Configuration (.env)
Le projet utilise des variables d'environnement qui seront stockés comme secret dans le Azure Key Vault.

## ▶️ Démarrage de l'application

Une fois tout installé, lancez l'interface Streamlit :

```bash 
streamlit run app/query.py
```

## 🐳 Démarrage rapide avec Docker 

Au lieu d'installer Python et Ollama manuellement, vous pouvez lancer tout le projet (Frontend + Backend) en une seule commande grâce à **Docker Compose**.

```bash 
docker-compose up --build
```

## 📂 Structure du projet

```text
Projet_Cloud_M2/
├── .github/
│   └── workflows/         # Configuration du pipeline CI GitHub Actions
│          ├── ci-test.yml #  Exec Test CI
├── app/                   # FRONTEND (Streamlit)
│   ├── Dockerfile         # Configuration de l'image Frontend
│   ├── query.py           # Interface principale & Logique RAG
│   ├── chunking.py        # Script de découpage et d'ingestion des PDF
│   ├── requirements.txt   # Dépendances Python
│   └── data_pdf/          # Dossier de stockage temporaire des PDF
├── backend_ollama/        # BACKEND (Ollama)
│   ├── Dockerfile         # Configuration de l'image Backend
│   └── entrypoint.sh      # Script d'installation des modèles dans l'image
├── tests/                 # TESTS UNITAIRES
    └── test_rag_pipeline.py # Tests de configuration
```


