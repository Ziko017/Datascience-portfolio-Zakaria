from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from sentence_transformers import CrossEncoder
from mistralai import Mistral
import uuid

from rag_pipeline import charger_ressource, tour_de_parole
from fastapi.middleware.cors import CORSMiddleware


# ===========
# App FastAPI
# ===========
app = FastAPI(title="IA Socratique Cybersécurité")

#Connecter l'api au même local host que notre front end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================
# Chargement ressources au démarrage
# ==================================
chunks, index_faiss, bm25, modele = charger_ressource()
reranker = CrossEncoder(
    "H:/huggingface_cache/bge-reranker-v2-m3", 
    max_length=512
)
mistral_client = Mistral(api_key="3xnVQPuDcNPV8xuVxO3GVuJE045l55hP")

# ===================
# Stockage en mémoire
# ===================
sessions      = {}   # session_id → session étudiant
credentials_db = {}  # username → {password_hash, expire_at}
leaderboard   = {}   # username → points totaux
badges_db     = {}   # username → liste badges
progression_db = {}  # username → {forensics: niveau, cvss: niveau}

monitoring = {
    "total_questions": 0,
    "fallback_count": 0,
    "scores_par_domaine": {
        "forensics": [],
        "cve_cvss": [],
        "logs": []
    },
    "questions_frequentes": {}
}

#=======================================
# Créer les classes/objets avec Pydantic
#=======================================
class CreerUtilisateur(BaseModel):
    username: str
    password: str
    duree_heures: int = 6   #Accès 6h par défaut

class CreerClasse(BaseModel):
    usernames: list[str] #liste de tous les étudiants
    password : str #Mdp commun

class Token(BaseModel):
    access_token: str
    token_type : str

class InitSession(BaseModel):
    niveau_declare: str  #Débutant /Intermédiaire/ avancé

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    reponse_etudiant : Optional[str] = None
    domaine: Optional[str] = None #forensics/cvss...
    demande_indice: bool = False   #Hint système

# Config JWT + Sécurité
# Header + Payload du JWT
SECRET_KEY  = "ia_socratique_secret_key_2024"
ALGORITHM   = "HS256"
PROF_SECRET = "prof_secret_2024"

#l'algorithme de hashage le plus sécurisé pour les mots de passe
pwd_context    = CryptContext(schemes=["bcrypt"], deprecated="auto")
#le système qui dit à FastAPI où trouver le token JWT dans les requêtes
oauth2_scheme  = OAuth2PasswordBearer(tokenUrl="login")

# =============
# Fonctions JWT
# =============
def creer_token(username : str, expire_heures: int = 6):
    expire = datetime.utcnow() + timedelta(hours= expire_heures) #heure actuelle en utc + durée de x heures
    data = {"sub": username, "exp": expire}
    return jwt.encode(data, SECRET_KEY, ALGORITHM)

def verifier_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalide"
            )
        return username  # ← on retire la vérification credentials_db
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expiré ou invalide"
        )
        
    
#======================
# Fonctions utilitaires
#======================
def verifier_badges(username: str, resultat: dict):
    """Vérifie et attribue les badges mérités"""
    
    nouveaux = []
    badges_actuels = badges_db.get(username, [])
    points = leaderboard.get(username, 0)
    
    badges_possibles = {
        "🥉 Premiers pas"  : points >= 1,
        "🥈 Investigateur" : resultat["niveau"] == "intermédiaire",
        "🥇 Expert"        : resultat["niveau"] == "avancé",
        "⚡ Flash"         : resultat["score"] == 3,
        "  Einstein"      : points >= 10
    }
    
    for badge, condition in badges_possibles.items():
        if condition and badge not in badges_actuels:
            badges_actuels.append(badge)
            nouveaux.append(badge)
    
    badges_db[username] = badges_actuels
    return nouveaux


# ===========
# Routes PROF
# ===========

@app.post("/prof/creer_session_classe")
def creer_session_classe(password: str, 
                          duree_heures: int = 3,
                          prof_key: str = ""):
    """
    Le prof crée un accès pour toute la classe
    avec un seul mot de passe commun
    Les étudiants choisissent leur propre username
    """
    if prof_key != PROF_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Accès refusé — clé prof incorrecte"
        )
    
    expire_at = datetime.utcnow() + timedelta(hours=duree_heures)
    
    # Stocker le mot de passe de session classe
    credentials_db["__session_classe__"] = {
        "password": password,
        "expire_at": expire_at
    }
    
    return {
        "message": "Session classe créée",
        "password": password,
        "expire_at": expire_at.isoformat(),
        "duree_heures": duree_heures
    }


@app.get("/prof/leaderboard")
def voir_leaderboard(prof_key: str = ""):
    """
    Le prof voit le classement en temps réel
    """
    if prof_key != PROF_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Accès refusé — clé prof incorrecte"
        )
    
    # Trier par points décroissants
    classement = sorted(
        leaderboard.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        "leaderboard": [
            {
                "rang": i + 1,
                "username": username,
                "points": points
            }
            for i, (username, points) in enumerate(classement)
        ]
    }


@app.get("/prof/monitoring")
def voir_monitoring(prof_key: str = ""):
    """
    Le prof voit les stats globales
    taux fallback, score moyen, questions fréquentes
    """
    if prof_key != PROF_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Accès refusé — clé prof incorrecte"
        )
    
    total = monitoring["total_questions"]
    fallback = monitoring["fallback_count"]
    
    return {
        "total_questions": total,
        "taux_fallback": round(fallback / total, 2) if total > 0 else 0,
        "questions_frequentes": dict(
            sorted(
                monitoring["questions_frequentes"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # top 10
        ),
        "heatmap_competences": {
            domaine: round(sum(scores) / len(scores), 2)
            if scores else 0
            for domaine, scores in monitoring["scores_par_domaine"].items()
        }
    }

#============
# Route LOGIN
#============

@app.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    L'étudiant se connecte avec :
    - username : son prénom/pseudo librement choisi
    - password : le mot de passe donné par le prof au tableau
    """
    
    # Vérifier que la session classe existe
    if "__session_classe__" not in credentials_db:
        raise HTTPException(
            status_code=401,
            detail="Aucune session classe active — contacte ton professeur"
        )
    
    session_classe = credentials_db["__session_classe__"]
    
    # Vérifier expiration
    if datetime.utcnow() > session_classe["expire_at"]:
        raise HTTPException(
            status_code=401,
            detail="Session expirée — contacte ton professeur"
        )
    
    # Vérifier le mot de passe
    if form.password != session_classe["password"]:
        raise HTTPException(
            status_code=401,
            detail="Mot de passe incorrect"
        )
    
    # Initialiser le profil étudiant si nouveau
    if form.username not in leaderboard:
        leaderboard[form.username]    = 0
        badges_db[form.username]      = []
        progression_db[form.username] = {
            "forensics": "débutant",
            "cve_cvss" : "débutant",
            "logs"     : "débutant"
        }
    
    # Créer le token JWT
    token = creer_token(form.username)
    
    return {
        "access_token": token,
        "token_type": "bearer"
    }

#===================
# Route SESSION/INIT
#===================

@app.post("/session/init")
def init_session(body: InitSession,
                 username: str = Depends(verifier_token)):
    """
    L'étudiant déclare son niveau et choisit son domaine
    Nécessite d'être connecté (token JWT)
    """
    
    session_id = str(uuid.uuid4())
    
    sessions[session_id] = {
        "username": username,
        "niveau_declare": body.niveau_declare,
        "niveau_estime": body.niveau_declare,
        "niveau_final": body.niveau_declare,
        "score": 0,
        "nb_echanges": 0,
        "calibration_terminee": False,
        "historique": [],
        "bloque": 0
    }
    
    return {
        "session_id": session_id,
        "username": username,
        "niveau_declare": body.niveau_declare,
        "message": f"Session créée — bonne chance {username} !"
    }

#===============
# Route QUESTION
#===============

@app.post("/question")
def poser_question(body: QuestionRequest,
                   username: str = Depends(verifier_token)):
    """
    L'étudiant pose une question et reçoit
    une réponse socratique adaptée à son niveau
    """
    
    # Vérifier que la session existe
    if body.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session introuvable — crée une session d'abord"
        )
    
    session = sessions[body.session_id]
    
    # Vérifier que la session appartient à cet étudiant
    if session.get("username") != username:
        raise HTTPException(
            status_code=403,
            detail="Session non autorisée"
        )
    
    # Gestion hint system
    if body.demande_indice:
        session["score"] = max(0, session["score"] - 1)
        session["bloque"] = 1
    
    # Tour de parole
    resultat = tour_de_parole(
        question=body.question,
        reponse_etudiant=body.reponse_etudiant,
        session=session,
        chunks=chunks,
        index_faiss=index_faiss,
        bm25=bm25,
        modele=modele,
        reranker=reranker,
        mistral_client=mistral_client
    )
    
    # Mettre à jour la session
    sessions[body.session_id] = resultat["session"]
    
    # Mettre à jour le leaderboard
    if resultat["score"] > session.get("score", 0):
        points_gagnes = resultat["score"] - session.get("score", 0)
        leaderboard[username] = leaderboard.get(username, 0) + points_gagnes
    
    # Mettre à jour la progression par domaine
    if body.domaine and username in progression_db:
        progression_db[username][body.domaine] = resultat["niveau"]
    
    # Mettre à jour le monitoring
    monitoring["total_questions"] += 1
    question_lower = body.question.lower()
    monitoring["questions_frequentes"][question_lower] = \
        monitoring["questions_frequentes"].get(question_lower, 0) + 1
    
    # Vérifier badges
    nouveaux_badges = verifier_badges(username, resultat)
    
    return {
        "question_socratique": resultat["question_socratique"],
        "feedback"           : resultat["feedback"],
        "score"              : resultat["score"],
        "niveau"             : resultat["niveau"],
        "mode"               : resultat["mode"],
        "points_total"       : leaderboard.get(username, 0),
        "nouveaux_badges"    : nouveaux_badges
    }


# ─────────────────────────────────────
# Route PROGRESSION
# ─────────────────────────────────────

@app.get("/progression")
def voir_progression(username: str = Depends(verifier_token)):
    """
    L'étudiant voit sa progression par domaine
    """
    
    progression = progression_db.get(username, {
        "forensics": "débutant",
        "cve_cvss" : "débutant",
        "logs"     : "débutant"
    })
    
    return {
        "username"   : username,
        "progression": progression,
        "points"     : leaderboard.get(username, 0),
        "badges"     : badges_db.get(username, [])
    }


# ─────────────────────────────────────
# Route BADGES
# ─────────────────────────────────────

@app.get("/badges")
def voir_badges(username: str = Depends(verifier_token)):
    """
    L'étudiant voit ses badges débloqués
    """
    
    badges_actuels = badges_db.get(username, [])
    
    # Tous les badges possibles avec statut
    tous_badges = {
        " Premiers pas"  : "débloqué" if " Premiers pas"  in badges_actuels else "🔒",
        " Investigateur" : "débloqué" if " Investigateur" in badges_actuels else "🔒",
        " Expert"        : "débloqué" if " Expert"        in badges_actuels else "🔒",
        " Flash"         : "débloqué" if " Flash"         in badges_actuels else "🔒",
        " Einstein"      : "débloqué" if " Einstein"      in badges_actuels else "🔒"
    }
    
    return {
        "username"    : username,
        "badges"      : tous_badges,
        "total_badges": len(badges_actuels)
    }


#==================
# Route LEADERBOARD
#==================

@app.get("/leaderboard")
def voir_leaderboard_public(username: str = Depends(verifier_token)):
    """
    Tous les étudiants voient le classement
    """
    
    classement = sorted(
        leaderboard.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Trouver le rang de l'étudiant connecté
    rang_etudiant = next(
        (i + 1 for i, (u, _) in enumerate(classement) if u == username),
        None
    )
    
    return {
        "leaderboard": [
            {
                "rang"    : i + 1,
                "username": u,
                "points"  : p,
                "moi"     : u == username  # ← highlight l'étudiant connecté
            }
            for i, (u, p) in enumerate(classement)
        ],
        "mon_rang": rang_etudiant
    }

# ─────────────────────────────────────
# Route DÉFI DU JOUR
# ─────────────────────────────────────
import json
from datetime import date
import random

def generer_defi_du_jour():
    """
    Génère un défi basé sur un chunk aléatoire
    Change chaque jour — même seed = même défi pour tout le monde
    """
    # Seed basé sur la date → même défi pour toute la classe
    random.seed(str(date.today()))
    chunk_defi = random.choice(chunks)
    
    prompt = f"""Tu es un expert en cybersécurité.
Génère un défi du jour basé sur ce contenu.
Le défi doit être un scénario concret et engageant.

Réponds UNIQUEMENT avec un JSON valide :
{{
    "titre": "titre court du défi",
    "scenario": "description du scénario en 2-3 phrases",
    "question": "question principale du défi",
    "domaine": "{chunk_defi['domain']}",
    "difficulte": "débutant ou intermédiaire ou avancé"
}}

Contenu :
{chunk_defi['content'][:500]}"""

    reponse = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    
    contenu = reponse.choices[0].message.content
    contenu = contenu.strip().replace("```json", "").replace("```", "").strip()
    
    try:
        defi = json.loads(contenu)
        defi["date"] = str(date.today())
        defi["chunk_id"] = chunk_defi["id"]
        return defi
    except:
        return {
            "titre"     : "Défi du jour",
            "scenario"  : "Analyse ce scénario forensics",
            "question"  : chunk_defi["content"][:200],
            "domaine"   : chunk_defi["domain"],
            "difficulte": "intermédiaire",
            "date"      : str(date.today()),
            "chunk_id"  : chunk_defi["id"]
        }


@app.get("/defi_du_jour")
def defi_du_jour(username: str = Depends(verifier_token)):
    """
    L'étudiant récupère le défi du jour
    Même défi pour toute la classe
    """
    defi = generer_defi_du_jour()
    return defi

#=============
# Route HEALTH
#=============

@app.get("/health")
def health():
    """
    Vérifie que l'API tourne correctement
    Utilisé par AWS ECS pour vérifier l'état du conteneur
    """
    return {
        "status"          : "ok",
        "chunks"          : len(chunks),
        "sessions_actives": len(sessions),
        "etudiants"       : len(leaderboard)
    }

#On lance uvicorn
if "__name__" == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)