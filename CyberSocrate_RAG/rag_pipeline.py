import numpy as np
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

###=========Charger les ressources============###
def charger_ressource():
    #Chunks
    chunks = []
    with open("Forensics.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    #Index_FAISS
    index_faiss = faiss.read_index("faiss_index.bin")
    #Index BM25
    with open("bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    #Modèle embedding
    modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    return chunks, index_faiss, bm25, modele


###==================Hybrid Search=======================###
def recherhe_hybride(question, chunks, index_faiss, bm25, modele,top_k=10, alpha =0.5):
    """
    Combine vector search (FAISS) et keyword search (BM25)
    
    alpha = poids du vecteur search
    alpha=0.5 → équilibre
    alpha=0.7 → favorise le sens sémantique
    alpha=0.3 → favorise les mots exacts
    """
    #1-Vector Search
    vecteur_question = modele.encode([question], convert_to_numpy = True)
    faiss.normalize_L2(vecteur_question)
    score_faiss, indice_faiss = index_faiss.search(vecteur_question,top_k)
    scores_faiss = score_faiss[0] #On a une seule question (FAISS considère qu'on pose pleins de questions du coup il fait un double crocchet)
    indices_faiss = indice_faiss[0]
    #Normalisation 0-1 (MinMax Scaler)
    if scores_faiss.max() > scores_faiss.min(): #éviter de diviser par 0
        scores_faiss = (scores_faiss - scores_faiss.min())/ (scores_faiss.max()-scores_faiss.min())

    #2-BM25 Search
    question_tokenize = question.lower().split()
    score_bm25 = bm25.get_scores(question_tokenize)
    #Normalisation 0-1 (MinMaxScaler)
    if score_bm25.max() > score_bm25.min():
        score_bm25 = (score_bm25 - score_bm25.min())/(score_bm25.max()- score_bm25.min())
    
    #3-Fusion: Vector Search + BM25 Search
    scores_hybrides = np.zeros(len(chunks)) # [0]*76
    for i, idx in enumerate(indices_faiss): #indice FAISS => ex:[2,4,5,8,6,3,9,8] => les positions des meilleurs chunks dans ta liste chunks
        scores_hybrides[idx] += alpha* scores_faiss[i]
    scores_hybrides += (1-alpha)* score_bm25

    #4-Top-K
    sorted_indices = np.argsort(scores_hybrides)
    top_indices = sorted_indices[::-1][:top_k] #Inverse la liste et prendre premiers topk élement

    resultats = []
    for indice in top_indices:
        resultats.append({
            "chunk":chunks[indice],
            "score_hybride": float(scores_hybrides[indice])
        })
    
    return resultats 



###======================BGE Reranker================================###
def reranker_chunks(question, resultats, reranker, top_k=3):
    """
    Reranke les résultats de la hybrid search
    en analysant la relation question-chunk
    
    resultats : liste retournée par recherche_hybride()
    top_k     : nombre de chunks à retourner après reranking
    """
    
    #Construire les paires (question, chunk_content)
    paires = [
        [question, r["chunk"]["content"]]
        for r in resultats
    ]
    
    #Calculer les scores BGE
    scores_bge = reranker.predict(paires)
    
    # Ajouter le score BGE à chaque résultat
    for i, r in enumerate(resultats):
        r["score_bge"] = float(scores_bge[i])
    
    # Trier par score BGE décroissant
    resultats_rerankés = sorted(
        resultats,
        key=lambda x: x["score_bge"],
        reverse=True
    )
    
    return resultats_rerankés[:top_k]


###====================Query Expansion=======================###
import os
from mistralai import Mistral
import json

#Clé API Mistral
mistral_client = Mistral(api_key="kzsns4JP59r7t4d7lmRBMJSF5Io1DcRv")

def expansion_question(question, mistral_client, n=3): #n=3 => Nombres de reformulation à générer
    reponse = mistral_client.chat.complete(
        model = "mistral-small-latest",
        messages =[{
            "role":"user", #role => Qui parle dans la conversation(user)
            "content": f"""Tu es un expert en cybersécurité.
Génère {n} reformulations techniques de cette question
UNIQUEMENT dans le domaine de la cybersécurité.
Réponds UNIQUEMENT avec un JSON valide, rien d'autre :
["reformulation1", "reformulation2", "reformulation3"]

Question : {question}"""
        }]
        )
    contenu = reponse.choices[0].message.content
    # Nettoyer les backticks (===> problème rencontré en production)
    contenu = contenu.strip()
    contenu = contenu.replace("```json", "").replace("```", "").strip()
    
    variantes = json.loads(contenu)
    return [question] + variantes


###==================PIPELINE RAG (CHEF D'ORCHESTRE)=========================###
def pipeline_rag(question, chunks, index_faiss, bm25, modele,
                 reranker, mistral_client, top_k=5, alpha=0.5):
    # Query Expansion supprimée
    # BGE Reranker supprimé
    
    # Hybrid Search directement
    resultats = recherhe_hybride(
        question, chunks, index_faiss, 
        bm25, modele, top_k=5, alpha=0.5
    )
    
    # Top 3 directs par score hybride
    top_3 = sorted(
        resultats,
        key=lambda x: x["score_hybride"],
        reverse=True
    )[:3]
    
    return top_3
    resultats_fusionnes = list(tous_les_résultats.values())
    #Reranker
    #top_3 = reranker_chunks(question,resultats_fusionnes,reranker,top_k=3)
    return top_3



#######################=================PROMPT SOCRATIQUE====================############################
def prompt_socratique_avance(question, reponse_etudiant, 
                              top_3_chunks, session):
    
    # Contexte RAG
    contexte = ""
    for i, r in enumerate(top_3_chunks):
        contexte += f"\n--- Source {i+1} ---\n"
        contexte += r["chunk"]["content"]
        contexte += "\n"
    
    # Historique
    historique_str = ""
    for echange in session["historique"][-3:]:
        historique_str += f"Question : {echange['question_socratique']}\n"
        historique_str += f"Réponse  : {echange['reponse_etudiant']}\n\n"
    
    # Phase calibration ou normal
    if not session["calibration_terminee"]:
        phase = f"""
PHASE DE CALIBRATION (échange {session['nb_echanges']}/3) :
L'étudiant a déclaré être niveau : {session['niveau_declare']}
Analyse son vocabulaire et la précision de ses réponses.
Ajuste le niveau estimé si nécessaire.
Après 3 échanges → mets calibration_terminee à true.
"""
    else:
        phase = f"""
NIVEAU STABILISÉ : {session['niveau_final']}
Calibration terminée — applique ce niveau strictement.
"""

    # Style selon score
    if session["score"] == 0:
        style = """Questions fondamentales sur les concepts cyber.
Pas d'analogies du quotidien — reste dans le domaine technique.
Ex: "Quelle différence entre une trace volatile et persistante ?"
"""
    elif session["score"] == 1:
        style = """Questions sur le "pourquoi" technique.
Pousse l'étudiant à expliquer les mécanismes.
Ex: "Pourquoi ce registre persiste après suppression du fichier ?"
"""
    elif session["score"] == 2:
        style = """Mise en situation forensique réelle.
L'étudiant doit appliquer le concept dans un scénario.
Ex: "Dans cette investigation, comment utiliserais-tu cet artefact ?"
"""
    
    # Cas score >= 3 — synthèse
    if session["score"] >= 3:
        reponses = "\n".join([
            f"- {e['reponse_etudiant']}" 
            for e in session["historique"]
        ])
        
        prompt = f"""Tu es un assistant pédagogique socratique en cybersécurité.
L'étudiant a atteint le score maximum.

RÉPONSES DE L'ÉTUDIANT :
{reponses}

Génère une synthèse de CE QU'IL A DÉCOUVERT par lui-même.
Félicite-le et indique le passage au niveau supérieur.

Réponds UNIQUEMENT avec ce JSON :
{{
    "question_socratique": "synthèse et félicitations",
    "niveau_estime": "{session['niveau_final']}",
    "score_delta": 0,
    "feedback": "message de félicitation",
    "calibration_terminee": true,
    "mode": "synthese",
    "nouveau_niveau": "{niveau_suivant(session['niveau_final'])}"
}}"""
        return prompt

    # Gestion "je sais pas"
    gestion_bloque = ""
    if session["bloque"] >= 2:
        gestion_bloque = """
ATTENTION : L'étudiant est bloqué depuis 2 échanges.
→ Donne un indice technique concret sans donner la réponse.
→ Reformule la question différemment.
→ mode = "indice_large"
"""
    elif session["bloque"] == 1:
        gestion_bloque = """
L'étudiant a dit ne pas savoir.
→ Donne un indice léger pour l'orienter.
→ Ne pénalise pas le score.
→ mode = "indice"
"""

    # Prompt standard
    prompt = f"""Tu es un assistant pédagogique socratique en cybersécurité.
Ton rôle : guider l'étudiant vers la réponse par le questionnement.
Ne jamais donner la réponse directement.

{phase}

PROFIL ÉTUDIANT :
- Niveau déclaré  : {session['niveau_declare']}
- Niveau estimé   : {session['niveau_estime']}
- Score           : {session['score']}/3
- Échanges        : {session['nb_echanges']}

HISTORIQUE :
{historique_str if historique_str else "Premier échange"}

STYLE DE QUESTIONNEMENT (score {session['score']}/3) :
{style}

{gestion_bloque}

RÈGLES ABSOLUES :
- UNE seule question à la fois
- Reste TOUJOURS dans le domaine cybersécurité
- Pas d'analogies du quotidien — vocabulaire technique
- Évalue la réponse et ajuste le niveau si nécessaire

CONTEXTE TECHNIQUE (utilise-le pour guider) :
{contexte}

QUESTION DE L'ÉTUDIANT : {question}
RÉPONSE DE L'ÉTUDIANT  : {reponse_etudiant if reponse_etudiant else "Premier échange — pas encore de réponse"}

Réponds UNIQUEMENT avec ce JSON valide :
{{
    "question_socratique": "ta question socratique ici",
    "niveau_estime": "débutant ou intermédiaire ou avancé",
    "score_delta": 0 ou 1,
    "feedback": "message court adapté à la réponse",
    "calibration_terminee": true ou false,
    "mode": "normal ou indice ou indice_large ou synthese"
}}"""

    return prompt


def niveau_suivant(niveau_actuel):
    """Retourne le niveau suivant"""
    niveaux = ["débutant", "intermédiaire", "avancé"]
    idx = niveaux.index(niveau_actuel)
    if idx < len(niveaux) - 1:
        return niveaux[idx + 1]
    return "expert"

def tour_de_parole(question, reponse_etudiant, session,
                   chunks, index_faiss, bm25, modele,
                   reranker, mistral_client):
    
    #RAG
    top_3 = pipeline_rag(
        question, chunks, index_faiss,
        bm25, modele, reranker, mistral_client
    )
    
    #Prompt socratique avancé
    prompt = prompt_socratique_avance(
        question, reponse_etudiant, top_3, session
    )
    
    # Appel Mistral
    reponse = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    
    contenu = reponse.choices[0].message.content
    contenu = contenu.strip().replace("```json", "").replace("```", "").strip()
    
    try:
        resultat = json.loads(contenu)
    except json.JSONDecodeError:
        resultat = {
            "question_socratique": "Peux-tu reformuler ta question ?",
            "niveau_estime": session["niveau_estime"],
            "score_delta": 0,
            "feedback": "",
            "calibration_terminee": session["calibration_terminee"],
            "mode": "normal"
        }
    
    # Mettre à jour la session
    session["score"] += resultat["score_delta"]
    session["niveau_estime"] = resultat["niveau_estime"]
    session["nb_echanges"] += 1
    session["calibration_terminee"] = resultat["calibration_terminee"]
    
    # Détecter si bloqué
    if reponse_etudiant and any(
        mot in reponse_etudiant.lower() 
        for mot in ["sais pas", "aucune idée", "comprends pas", "?"]
    ):
        session["bloque"] += 1
    else:
        session["bloque"] = 0
    
    # Niveau final = niveau estimé après calibration
    if resultat["calibration_terminee"]:
        session["niveau_final"] = resultat["niveau_estime"]
    
    # Monter de niveau si score >= 3
    if session["score"] >= 3:
        session["niveau_final"] = niveau_suivant(session["niveau_final"])
        session["niveau_estime"] = session["niveau_final"]
        session["score"] = 0
    
    # Ajouter à l'historique
    session["historique"].append({
        "question_socratique": resultat["question_socratique"],
        "reponse_etudiant": reponse_etudiant or ""
    })
    
    return {
        "question_socratique": resultat["question_socratique"],
        "feedback": resultat.get("feedback", ""),
        "score": session["score"],
        "niveau": session["niveau_final"],
        "mode": resultat.get("mode", "normal"),
        "session": session
    }







    

