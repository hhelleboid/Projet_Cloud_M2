import streamlit as st
import time
import os
import json 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from chunking import process_all_documents 


# Sur Azure, on définira cette variable à "/data" via la config du conteneur.
BASE_PERSIST_PATH = os.getenv("PERSIST_DIRECTORY", ".")

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

# --- 1. CONFIGURATION & CONSTANTES ---
class Config:
    # CHROMA_PATH = "chromadb"
    # DATA_PATH = "data_pdf" # Nom du dossier contenant les PDFs
    CHROMA_PATH = os.path.join(BASE_PERSIST_PATH, "chromadb")
    DATA_PATH = os.path.join(BASE_PERSIST_PATH, "data_pdf")
    
    RERANKER_PATH = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    EMBEDDING_MODEL = "nomic-embed-text"
    DEFAULT_LLM_MODEL = "gemma3:1b"
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434") # # URL du serveur LLM (autre conteneur), ne pas oublier de mettre les 2 containers sur le même réseau 

PROMPT_TEMPLATE = """Tu es un assistant francophone strictement ancré au contexte fourni.
Réponds en français, de façon concise et exacte.
RÈGLES:
- Réponds UNIQUEMENT avec le contexte ci-dessous.
- Si l'information n'est pas présente, dis: "Non trouvé dans le contexte fourni."
- Cite les éléments numériques exactement.
- Ne réponds pas plus que nécessaire.

Contexte:
{context}

Question: {question}

Réponse:"""


# --- GESTION DE LA PERSISTANCE ---
HISTORY_FILE = os.path.join(BASE_PERSIST_PATH, "chat_history.json")

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

# --- 2. CHARGEMENT DES RESSOURCES (CACHÉ) ---
# Cette fonction ne s'exécute qu'une seule fois au démarrage
@st.cache_resource
def load_resources():
    print("🔄 Chargement des modèles en mémoire...")
    start_load = time.perf_counter()
    
    # 1. Embeddings
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL, base_url=Config.LLM_BASE_URL)
    
    # 2. ChromaDB
    db = Chroma(
        embedding_function=embeddings,
        collection_name="pdf_docs",
        persist_directory=Config.CHROMA_PATH
    )
    
    # 3. Reranker
    try:
        reranker = CrossEncoder(Config.RERANKER_PATH, device="cpu")
    except Exception as e:
        st.warning(f"Modèle local non trouvé, le modèle sera téléchargé depuis HuggingFace. Détails techniques:  ({e})")
        reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu")  # sauvegarde auto dans le cache sinon
    
    print(f"✅ Chargement terminé en {time.perf_counter() - start_load:.2f}s")
    return db, reranker

# Chargement initial
try:
    vector_db, reranker_model = load_resources()
except Exception as e:
    st.error(f"Erreur critique au chargement des ressources : {e}")
    st.stop()

# --- 3. INTERFACE UTILISATEUR ---

st.title("Assistant Documentaire PDF ")

# Vérification de la présence de fichiers PDF
if not os.path.exists(Config.DATA_PATH):
    st.warning(f"⚠️ Le dossier '{Config.DATA_PATH}' est introuvable.")
else:
    pdf_files = [f for f in os.listdir(Config.DATA_PATH) if f.lower().endswith('.pdf')]
    if not pdf_files:
        st.warning(f"⚠️ Aucun fichier PDF trouvé dans le dossier '{Config.DATA_PATH}'. Veuillez ajouter des documents pour que l'assistant puisse répondre.")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    llm_model = st.selectbox(
        "Modèle Ollama au préalable téléchargé", 
        ["gemma3:1b"],
        index=0
    )
    
    k_candidates = st.slider("Candidats (Retrieval)", 10, 100, 30)
    top_k_final = st.slider("Top K (Reranking)", 1, 10, 3)
    
    st.divider()
    st.info("Le Reranking améliore la précision mais ralentit la réponse.")

    # --- AJOUT : LISTE DES PDFS ---
    st.divider()
    st.header("📂 Documents indexés")
    
    if os.path.exists(Config.DATA_PATH):
        files_in_folder = [f for f in os.listdir(Config.DATA_PATH) if f.lower().endswith('.pdf')]
        
        if files_in_folder:
            st.success(f"{len(files_in_folder)} fichiers disponibles")
            with st.expander("Voir la liste détaillée"):
                for f in files_in_folder:
                    st.caption(f"📄 {f}")
        else:
            st.warning("Aucun fichier PDF trouvé.")
    else:
        st.error(f"Dossier '{Config.DATA_PATH}' introuvable.")
        
        
    # --- AJOUT : UPLOAD DE PDFS ---
        
    st.divider()
    st.header("📤 Ajouter des documents")
    
    uploaded_files = st.file_uploader(
        "Déposez vos fichiers PDF ici", 
        type=['pdf'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        if not os.path.exists(Config.DATA_PATH):
            os.makedirs(Config.DATA_PATH)
        
        new_files_count = 0
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(Config.DATA_PATH, uploaded_file.name)
            # On vérifie si le fichier existe déjà pour éviter de l'écraser inutilement
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_count += 1
                st.success(f"✅ '{uploaded_file.name}' sauvegardé !")
                
                
        # Si de nouveaux fichiers ont été ajoutés, on lance le chunking
        if new_files_count > 0:
            with st.status("⚙️ Indexation des nouveaux documents...", expanded=True) as status:
                st.write("Analyse et découpage des PDF...")
                
                # 1. Lancer le processus de chunking
                process_all_documents()
                
                st.write("Mise à jour de la base de données...")
                
                # 2. Vider le cache de Streamlit pour forcer le rechargement de la DB
                st.cache_resource.clear()
                
                # 3. Recharger les ressources immédiatement pour la session en cours
                vector_db, reranker_model = load_resources()
                
                status.update(label="✅ Indexation terminée ! Vous pouvez poser vos questions.", state="complete", expanded=False)
                time.sleep(1) # Petit temps pour voir le message vert
                st.rerun() # Rafraîchit la page pour afficher la nouvelle liste de fichiers
                
    st.divider()
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        save_chat_history([])
        st.rerun()
                

# Initialisation de l'historique de chat
# if "messages" not in st.session_state:
#     st.session_state.messages = []
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history() 

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if query_text := st.chat_input("Posez votre question sur les documents..."):
    
    # Affichage message utilisateur
    st.session_state.messages.append({"role": "user", "content": query_text})
    save_chat_history(st.session_state.messages) 
    
    with st.chat_message("user"):
        st.markdown(query_text)

    # Traitement Assistant
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Utilisation de st.status pour montrer la progression
        with st.status("Analyse des documents en cours...", expanded=True) as status:
            
            start_total = time.perf_counter()
            
            # On peut rajouter une étape de reformulation ici si besoin
            
            # --- ETAPE 1 : RETRIEVAL ---
            st.write("🔍 Recherche vectorielle ...")
            start_retrieval = time.perf_counter()
            # Utilisation de similarity_search pour la vitesse
            candidates = vector_db.similarity_search(query_text, k=k_candidates)
            time_retrieval = time.perf_counter() - start_retrieval
            st.write(f"✅ {len(candidates)} documents trouvés ({time_retrieval:.4f}s)")
            
            # --- ETAPE 2 : RERANKING ---
            st.write("🔍 ReRanking...")
            start_rerank = time.perf_counter()
            
            if candidates:
                pairs = [[query_text, doc.page_content] for doc in candidates]
                scores = reranker_model.predict(pairs)
                scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                final_docs = [(doc, score) for doc, score in scored_docs[:top_k_final]]
            else:
                final_docs = []
                
            time_rerank = time.perf_counter() - start_rerank
            st.write(f"✅ Top {len(final_docs)} sélectionnés ({time_rerank:.4f}s)")
            
            # Affichage des sources dans le status (expandable)
            with st.expander("Voir les documents sources utilisés"):
                for i, (doc, score) in enumerate(final_docs, 1):
                    st.markdown(f"**Doc {i} (Score: {score:.4f})**")
                    st.caption(f"ID: {doc.metadata.get('id', 'N/A')}")
                    st.text(doc.page_content[:300] + "...") # Aperçu
            
            # --- ETAPE 3 : GENERATION ---
            st.write(" Génération de la réponse...")
            start_gen = time.perf_counter()
            
            # Préparation du contexte (tronqué pour éviter overflow)
            context_text = "\n\n---\n\n".join([doc.page_content[:1500] for doc, _score in final_docs])
            
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            
            llm = OllamaLLM(
                model=llm_model,
                base_url=Config.LLM_BASE_URL,
                num_thread=4,
                num_ctx=2048, # Sécurité mémoire
                temperature=0.1
            )
            
            response_text = llm.invoke(prompt)
            time_gen = time.perf_counter() - start_gen
            st.write(f"✅ Réponse générée ({time_gen:.4f}s)")
            
            # Mise à jour du statut final
            total_time = time.perf_counter() - start_total
            status.update(label=f"Réponse générée en {total_time:.2f}s", state="complete", expanded=False)

        # Affichage de la réponse finale
        st.markdown(response_text)
        
        # Sauvegarde dans l'historique
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
#  streamlit run query.py