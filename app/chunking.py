import fitz
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
from statistics import median
from copy import deepcopy

import argparse
import os
import shutil
from pathlib import Path
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # import actuel
import fitz  # PyMuPDF
import re
from collections import defaultdict


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")


CHROMA_PATH = "chromadb"
DATA_PATH = "data_pdf"
SPLIT_REGEX = re.compile(r"\.(?:\n\s*\n+|\n)")  # '.' puis '\n' OU '.' puis '\n' + blancs + '\n+'
# Découpe approximative en phrases : ponctuation forte suivie d'espaces
RE_SENT = re.compile(r"(?<=[\.!\?…])\s+")

def text_with_visual_linebreaks(page, line_gap_mult=0.6, para_gap_mult=1.2):
    """
    Reconstruit le texte d'une page en insérant des sauts de ligne visuels.
    - line_gap_mult : seuil relatif pour un retour à la ligne
    - para_gap_mult : seuil relatif pour un changement de paragraphe
    """
    data = page.get_text("dict")
    lines = []

    # Récupère les lignes texte (type=0) et leurs bbox
    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            # BBox de la ligne
            y_top = line["bbox"][1]
            y_bottom = line["bbox"][3]
            # Texte de la ligne : concat spans triés par x
            spans_sorted = sorted(spans, key=lambda s: s.get("bbox", [0,0,0,0])[0])
            parts = []
            last_rx = None
            t = []
            for sp in spans_sorted:
                txt = sp.get("text","")
                if not txt:
                    continue
                sxb, _, srx, _ = sp.get("bbox", [0,0,0,0])
                if last_rx is not None and sxb - last_rx > 1.0 and (not parts or not parts[-1].endswith(" ")):
                    parts.append(" ")
                parts.append(txt)
                last_rx = srx
                t.append(txt)
            # print("parts:", t)
            line_txt = " ".join("".join(parts).split())
            #print(f"Line text: {repr(line_txt)}")
            # line_txt = "".join("".join(t))
            if line_txt:
                # Stocker aussi les métadonnées de style pour détecter les titres
                avg_size, is_bold, is_ul = _avg_font_bold_ul(line)
                # print(f"Line: {repr(line_txt)} | avg_size: {avg_size:.1f} | bold: {is_bold}")
                lines.append((y_top, y_bottom, line_txt, avg_size, is_bold, is_ul, line))

    if not lines:
        return []

    # Tri vertical
    lines.sort(key=lambda t: (round(t[0],2), round(t[1],2)))
    #lines.sort(key=lambda t: (t[0], t[1]))


    # Hauteur de ligne médiane (robuste)
    heights = [yb - yt for yt, yb, _, _, _, _, _ in lines]
    line_h = median(heights) if heights else 10.0
    print(f"Hauteur médiane de ligne: {line_h}")
    print(f"Moyenne des hauteurs de ligne: {sum(heights) / len(heights) if heights else 0:.1f}")
    # Seuils pour détecter les ruptures
    line_gap = line_gap_mult * line_h
    para_gap = para_gap_mult * line_h

    out = []
    prev_bottom = None
    for i, (yt, yb, txt, avg_size, is_bold, is_ul, line_obj) in enumerate(lines):
        print(f"Line: {repr(txt)} | avg_size: {avg_size:.1f} | bold: {is_bold}")
        line_info = {
            "text": txt,
            "avg_size": avg_size,
            "is_bold": is_bold,
            "is_ul": is_ul,
            "line_obj": line_obj,
            "break_type": "none"
        }
        
        if i == 0:
            out.append(line_info)
            prev_bottom = yb
            continue

        gap = yt - prev_bottom

        # si fin de ligne précédente = tiret, recoller le mot
        if gap >= line_gap and gap < para_gap and out and out[-1]["text"].rstrip().endswith("-"):
            out[-1]["text"] = out[-1]["text"].rstrip()[:-1] + txt.lstrip()
            # print("Hyphen join detected")
            line_info["break_type"] = "hyphen_join"
        if gap >= para_gap:
            # print("Paragraph break detected", gap, para_gap)
            line_info["break_type"] = "paragraph"
            out.append(line_info)
        elif gap >= line_gap:
        # elif line_gap <= gap < para_gap:
            # print("Line break detected", gap, line_gap)
            line_info["break_type"] = "line"
            out.append(line_info)
        else:
            # même ligne visuelle - concaténer avec espace
            # print("Continuation detected", out[-1]["text"], " + ", txt, line_info["is_bold"])
            out.append(line_info)
            line_info["break_type"] = "continuation"

        prev_bottom = yb

    return out

def _concat_line_text(line) -> str:
    return " ".join(span.get("text", "") for span in line.get("spans", []))

def _normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()

def _avg_font_bold_ul(line) -> Tuple[float, bool, bool]:
    """Moyenne pondérée (par nb de caractères visibles), ignore les spans whitespace."""
    spans = line.get("spans", []) or []
    total_chars, weighted, bold, ul = 0, 0.0, False, False
    for s in spans:
        txt = s.get("text", "") or ""
        size = float(s.get("size", 0.0))
        flags = int(s.get("flags", 0))
        bold = bold or bool(flags & (2**4))       # gras
        ul   = ul   or bool(flags & (2**3))       # souligné
        if not txt or txt.isspace():
            continue
        n = len(txt)
        total_chars += n
        weighted += size * n
    if total_chars > 0:
        return weighted / total_chars, bold, ul
    # fallback si tout est whitespace: prend la plus grande taille vue
    return (max((float(s.get("size", 0.0)) for s in spans), default=0.0), bold, ul)

def _most_common_body_size(size_samples: List[float]) -> float:
    """Mode sur tailles arrondies au 0.5 pt, fallback 10.0 si vide."""
    if not size_samples:
        return 10.0
    buckets = [round(s * 2) / 2 for s in size_samples]
    return Counter(buckets).most_common(1)[0][0]

def extract_hierarchical_chunks(
    pdf_path: str,
    title_delta_pt: float = 1.0,          # écart min au corps pour considérer 'titre'
    require_bold_for_title: bool = True, # si True, un titre doit être gras
    accept_underline_as_title: bool = False,
    preserve_visual_linebreaks: bool = True  # nouveau paramètre
) -> List[Dict[str, Any]]:
    """
    Version avec text_with_visual_linebreaks pour préserver la mise en page.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    # -------- 1) Premier passage : collecte pour estimer body_size ----------
    size_samples: List[float] = []
    pages_lines = []
    
    for page in doc:
        if preserve_visual_linebreaks:
            lines_info = text_with_visual_linebreaks(page)
            pages_lines.append(lines_info)
            # Échantillonner pour body_size
            for line_info in lines_info:
                if not line_info["is_bold"] and line_info["avg_size"] > 5:
                    size_samples.append(line_info["avg_size"])
        else:
            # Ancien comportement
            blocks = page.get_text("dict").get("blocks", [])
            pages_lines.append(blocks)
            for b in blocks:
                for line in b.get("lines", []):
                    text = _normalize_space(_concat_line_text(line))
                    if not text:
                        continue
                    avg, is_bold, is_ul = _avg_font_bold_ul(line)
                    if not is_bold and avg > 5:
                        size_samples.append(avg)

    body_size = _most_common_body_size(size_samples)

    # -------- 2) Second passage : extraction par sections -------------------
    hierarchy = defaultdict(list)
    chunks: List[Dict[str, Any]] = []

    current_section = "Section 1"
    prev_section = current_section
    chunk: Dict[str, Any] = {
        "page_content": "",
        "metadata": {
            "total_pages": total_pages,
            "current_page": 1,
            "section": current_section,
            "source": str(pdf_path),
            "body_size_pt": body_size,
        }
    }

    for page_num, page_data in enumerate(pages_lines, start=1):
        if preserve_visual_linebreaks:
            # Traiter les lignes avec info visuelle
            for line_info in page_data:
                text = line_info["text"]
                avg = line_info["avg_size"]
                is_bold = line_info["is_bold"]
                is_ul = line_info["is_ul"]
                break_type = line_info["break_type"]
                
                if not text:
                    continue

                # Règle de titre
                style_ok = (is_bold or (accept_underline_as_title and is_ul)) if require_bold_for_title \
                           else (is_bold or (accept_underline_as_title and is_ul) or True)
                # is_title = (avg > body_size ) and is_bold
                is_title = (avg > body_size ) and style_ok

                if is_title:
                    current_section = text
                    _ = hierarchy[current_section]
                else:
                    # Ajouter le texte avec le bon séparateur selon break_type
                    if break_type == "paragraph":
                        line_text = "\n\n" + text
                    elif break_type == "line":
                        line_text = "\n" + text
                    else:
                        line_text = text if not chunk["page_content"] else " " + text

                    hierarchy[current_section].append(line_text)

                    if prev_section != current_section:
                        if chunk["page_content"]:
                            chunks.append(chunk)
                        chunk = {
                            "page_content": line_text.lstrip(),
                            "metadata": {
                                "total_pages": total_pages,
                                "current_page": page_num,
                                "section": current_section,
                                "source": str(pdf_path),
                                "body_size_pt": body_size,
                            }
                        }
                        prev_section = current_section
                    else:
                        chunk["page_content"] += line_text
                        chunk["metadata"]["current_page"] = page_num
                        chunk["metadata"]["section"] = current_section
        else:
            # Ancien comportement sans visual linebreaks
            blocks = page_data
            for b in blocks:
                for line in b.get("lines", []):
                    text = _normalize_space(_concat_line_text(line))
                    if not text:
                        continue

                    avg, is_bold, is_ul = _avg_font_bold_ul(line)
                    style_ok = (is_bold or (accept_underline_as_title and is_ul)) if require_bold_for_title \
                               else (is_bold or (accept_underline_as_title and is_ul) or True)
                    # is_title = (avg > body_size) and is_bold 
                    is_title = (avg > body_size ) and style_ok

                    if is_title:
                        current_section = text
                        _ = hierarchy[current_section]
                    else:
                        line_text = text if text else "\n"
                        hierarchy[current_section].append(line_text)

                        if prev_section != current_section:
                            if chunk["page_content"]:
                                chunks.append(chunk)
                            chunk = {
                                "page_content": line_text,
                                "metadata": {
                                    "total_pages": total_pages,
                                    "current_page": page_num,
                                    "section": current_section,
                                    "source": str(pdf_path),
                                    "body_size_pt": body_size,
                                }
                            }
                            prev_section = current_section
                        else:
                            sep = "\n" if chunk["page_content"] else ""
                            chunk["page_content"] += (sep + line_text)
                            chunk["metadata"]["current_page"] = page_num
                            chunk["metadata"]["section"] = current_section

    # -------- 3) push du dernier chunk si non vide --------------------------
    if chunk.get("page_content"):
        chunks.append(chunk)

    return chunks

def split_chunks_by_regex(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Découpe chaque chunk sur:
      - '.' suivi de '\n'
      - '.' suivi de '\n' + espaces (facult.) + '\n+'  (lignes blanches)
    En recréant des sous-chunks qui héritent des mêmes metadata.
    """
    out: List[Dict[str, Any]] = []

    for parent_idx, chunk in enumerate(chunks):
        text = chunk.get("page_content", "")
        meta = chunk.get("metadata", {}) or {}

        # On découpe mais on veut conserver le '.' terminal dans chaque sous-texte.
        # La regex split supprime le séparateur, donc on re-colle un '.' à la fin de chaque morceau,
        # sauf s'il est déjà présent (ex: texte fini par '?' ou ':' etc.).
        parts = SPLIT_REGEX.split(text)
        

        # Si le texte se termine par un point suivi d'un/des retours à la ligne,
        # split() produira une chaîne vide finale -> on l’enlève.
        if parts and parts[-1].strip() == "":
            parts = parts[:-1]

        # Nettoyage léger et reconstruction avec le '.' quand c'était un point séparateur
        # Astuce: on vérifie si l'original avait un point juste avant la coupure,
        # mais comme on split sur '.', on remet un '.' à la fin de chaque part
        # sauf la dernière si le texte original ne se terminait pas par un séparateur.
        for sub_idx, part in enumerate(parts):
            sub_text = part.strip()

            if not sub_text:
                continue

            # Ajoute un '.' de terminaison si le morceau n’en a pas déjà un raisonnable.
            # (Si la phrase se termine déjà par . ! ? … on n'ajoute rien.)
            if not re.search(r"[\.!\?…]$", sub_text):
                sub_text = sub_text + "."

            sub_chunk = {
                "page_content": sub_text,
                "metadata": deepcopy(meta)
            }

            # (Optionnel) rattacher des infos de provenance
            sub_chunk["metadata"]["parent_index"] = parent_idx
            sub_chunk["metadata"]["sub_index"] = sub_idx

            out.append(sub_chunk)

    return out


def _split_with_positions(text: str, regex: re.Pattern) -> List[Tuple[str, int, int]]:
    """
    Split 'text' avec 'regex' en conservant (segment, start, end) positions dans le texte d'origine.
    """
    parts: List[Tuple[str, int, int]] = []
    last = 0
    for m in regex.finditer(text):
        s, e = m.span()
        if s > last:
            parts.append((text[last:s], last, s))
        last = e
    if last < len(text):
        parts.append((text[last:], last, len(text)))
    return parts

def _greedy_pack(parts: List[Tuple[str, int, int]], max_chars: int) -> List[List[Tuple[str, int, int]]]:
    """
    Empaquette des phrases (texte, start, end) en blocs <= max_chars (ordre préservé).
    On recolle avec un espace unique entre phrases.
    """
    blocks: List[List[Tuple[str, int, int]]] = []
    cur: List[Tuple[str, int, int]] = []
    cur_len = 0
    for t, s, e in parts:
        t_stripped = t.strip()
        if not t_stripped:
            continue
        add = len(t_stripped) if not cur else (1 + len(t_stripped))  # +1 pour espace entre phrases
        if cur and cur_len + add > max_chars:
            blocks.append(cur)
            cur = [(t_stripped, s, e)]
            cur_len = len(t_stripped)
        else:
            if cur:
                cur_len += 1 + len(t_stripped)
            else:
                cur_len = len(t_stripped)
            cur.append((t_stripped, s, e))
    if cur:
        blocks.append(cur)
    return blocks

def _flatten(blocks: List[List[Tuple[str, int, int]]]) -> List[Tuple[str, int, int]]:
    """
    Chaque bloc -> (texte_recollé, start_min, end_max)
    """
    out: List[Tuple[str, int, int]] = []
    for b in blocks:
        txt = " ".join(x[0] for x in b)
        start = b[0][1]
        end = b[-1][2]
        out.append((txt, start, end))
    return out

def strict_slice(txt: str, s_abs: int, e_abs: int, max_chars: int) -> List[Tuple[str, int, int]]:
        res = []
        i = 0
        while i < len(txt):
            j = min(i + max_chars, len(txt))
            res.append((txt[i:j].strip(), s_abs + i, s_abs + j))
            i = j
        return res

def split_paragraph_chunks_by_sentence_length(
    chunks: List[Dict[str, Any]],
    max_chars: int = 1000,
    hard_upper_bound: int = 1200,
) -> List[Dict[str, Any]]:
    """
    On suppose que chaque chunk d'entrée est un paragraphe.
    - On découpe en phrases
    - On packe en sous-chunks <= max_chars (glouton)
    - Si un 'bloc' dépasse hard_upper_bound (phrase gigantesque), on tronque strictement
    """
    out: List[Dict[str, Any]] = []

    for parent_idx, ch in enumerate(chunks):
        text = ch.get("page_content", "") or ""
        if not text.strip():
            continue
        meta = deepcopy(ch.get("metadata", {}) or {})

        # 1) phrases + positions
        sents = _split_with_positions(text, RE_SENT)

        # 2) packing par longueur
        blocks = _greedy_pack(sents, max_chars)
        flats = _flatten(blocks)

        # 3) garde-fou + émission des sous-chunks
        for txt, s_abs, e_abs in flats:
            if len(txt) <= hard_upper_bound:
                out.append({
                    "page_content": txt,
                    "metadata": {
                        **deepcopy(meta),
                        "parent_index": parent_idx,
                        # sub_index local (par paragraphe)
                        "sub_index": sum(1 for c in out if c["metadata"]["parent_index"] == parent_idx),
                        "char_start": s_abs,
                        "char_end": e_abs,
                    }
                })
            else:
                # phrase/bloc exceptionnellement long -> coupe stricte
                for piece, ps, pe in strict_slice(txt, s_abs, e_abs, max_chars):
                    out.append({
                        "page_content": piece,
                        "metadata": {
                            **deepcopy(meta),
                            "parent_index": parent_idx,
                            "sub_index": sum(1 for c in out if c["metadata"]["parent_index"] == parent_idx),
                            "char_start": ps,
                            "char_end": pe,
                        }
                    })

    return out

def lister_fichiers(dossier):
    """Retourne la liste de tous les fichiers dans un dossier"""
    fichiers = []
    for root, _, files in os.walk(dossier):
        for file in files:
            fichiers.append(Path(root) / file)
    return fichiers


def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",       # ne pas oublier d'installer le modele nomic-embed-text sur ollama
        base_url=LLM_BASE_URL
    )
    return embeddings

def IDs_for_chunks(chunks: list[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("current_page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1

        else:
            current_chunk_index = 0
        
        last_page_id = current_page_id
        chunk.metadata["id"] = current_page_id + f":{current_chunk_index}"
        
    return chunks

def add_to_vectorstore(chunks: list[Document]):
    
    # charge la base de données chroma
    db = Chroma(
        embedding_function=get_embedding_function(),
        collection_name="pdf_docs",
        persist_directory=CHROMA_PATH
    )
    
    # ajoute des IDs aux chunks  
    chunks_ids = IDs_for_chunks(chunks)
    
    # ajouter ou mettre à jour les documents
    existing_items = db.get(include=[])  # IDs tjrs par défauts
    existing_ids = set(existing_items["ids"])
    print(f"Nombre de documents dans la DB: {len(existing_ids)}")
    
    # Seulement ajouter les documents qui n'existent pas dans la DB.
    new_chunks = []
    for chunk in chunks_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            
    if len(new_chunks):
        print(f"Ajout de nouveaux documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("Aucun nouveau document à ajouter")


def delete_from_vectorstore(name: str):
    db = Chroma(
        embedding_function=get_embedding_function(),
        collection_name="pdf_docs",
        persist_directory=CHROMA_PATH
    )
    
    to_delete = db.get(where={"source": f"data\{name}"}, include=[])["ids"]
    print("À supprimer:", len(to_delete), ":", to_delete)
    
    if to_delete:
        db.delete(ids=to_delete)
        print(f"Supprimé {len(to_delete)} item(s).")
    else:
        print("Aucun item trouvé.")

def clear_vectorstore():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Base de données vectorielle supprimée.")
    else:
        print("Aucune base de données vectorielle trouvée à supprimer.")

def add_context_to_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrichit le texte avec le contexte de la section"""
    enriched = []
    for chunk in chunks:
        section = chunk["metadata"]["section"]
        source = chunk["metadata"]["source"]
        text = chunk["page_content"]
        
        # Ajoute la section au début (sera vectorisée)
        enriched_text = f"Section: {section}| Source: {source}| {text}"
        
        enriched.append({
            "page_content": enriched_text,
            "metadata": chunk["metadata"]
        })
    
    return enriched
    
    
def process_all_documents():
    """Fonction principale pour lancer tout le processus d'indexation"""
    print(" Démarrage de l'indexation automatique...")
    
    # Lister tous les fichiers PDF
    all_files = lister_fichiers(DATA_PATH)
    if not all_files:
        print("Aucun fichier à traiter.")
        return

    # 1. Extraction hiérarchique
    hierarchical_chunks = [extract_hierarchical_chunks(str(fichier)) for fichier in all_files]
    
    # 2. Découpe par regex
    regex_split_chunks = [split_chunks_by_regex(chunk) for chunk in hierarchical_chunks]
    
    # 3. Découpe par longueur de phrases
    final_chunks = [
        split_paragraph_chunks_by_sentence_length(chunk, max_chars=1600, hard_upper_bound=1800)
        for chunk in regex_split_chunks
    ]
    
    # 4. Conversion en Documents LangChain
    docs = [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for chunk in final_chunks
        for c in chunk
    ]
    
    # 5. Ajout à la base vectorielle
    add_to_vectorstore(docs)
    print("✅ Indexation terminée avec succès !")

if __name__ == "__main__":
    # On peut maintenant simplement appeler la fonction
    process_all_documents()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
