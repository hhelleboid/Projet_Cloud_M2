import streamlit as st
import time
import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

# --- 1. CONFIGURATION & CONSTANTES ---
class Config:
    CHROMA_PATH = "chromadb"
    RERANKER_PATH = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    EMBEDDING_MODEL = "nomic-embed-text"
    # On laisse le choix du modèle LLM dans l'interface
    DEFAULT_LLM_MODEL = "mistral:7b-instruct-q5_K_M"
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://stupefied_curie:11434")

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

# --- 2. CHARGEMENT DES RESSOURCES (CACHÉ) ---
# Cette fonction ne s'exécute qu'une seule fois au démarrage
@st.cache_resource
def load_resources():
    print("🔄 Chargement des modèles en mémoire...")
    start_load = time.perf_counter()
    
    # 1. Embeddings
    OLLAMA_HOST = "http://stupefied_curie:11434"
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL,base_url=OLLAMA_HOST)
    
    # 2. ChromaDB
    db = Chroma(
        embedding_function=embeddings,
        collection_name="pdf_docs",
        persist_directory=Config.CHROMA_PATH
    )
    
    # 3. Reranker
    reranker = CrossEncoder(Config.RERANKER_PATH, device="cpu")
    
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

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    llm_model = st.selectbox(
        "Modèle Ollama au préalable téléchargé", 
        ["mistral:7b-instruct-q5_K_M", "llama3.2:3b", "mistral:7b"],
        index=0
    )
    
    k_candidates = st.slider("Candidats (Retrieval)", 10, 100, 30)
    top_k_final = st.slider("Top K (Reranking)", 1, 10, 3)
    
    st.divider()
    st.info("Le Reranking améliore la précision mais ralentit la réponse.")

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if query_text := st.chat_input("Posez votre question sur les documents..."):
    
    # Affichage message utilisateur
    st.session_state.messages.append({"role": "user", "content": query_text})
    with st.chat_message("user"):
        st.markdown(query_text)

    # Traitement Assistant
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Utilisation de st.status pour montrer la progression
        with st.status("Analyse des documents en cours...", expanded=True) as status:
            
            start_total = time.perf_counter()
            
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
                base_url=""http://stupefied_curie:11434"",
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
