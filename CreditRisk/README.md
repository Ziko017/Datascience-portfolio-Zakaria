# CreditRisk-ML-Drift-Analysis

Projet de Machine Learning appliqué à la prédiction du risque de crédit avec analyse et mitigation du data drift.

---

## 📋 Table des Matières

- [Description du Projet](#description-du-projet)
- [Objectifs](#objectifs)
- [Dataset](#dataset)
- [Méthodologie](#méthodologie)
- [Résultats Clés](#résultats-clés)
- [Structure du Projet](#structure-du-projet)
- [Technologies Utilisées](#technologies-utilisées)
- [Installation et Exécution](#installation-et-exécution)
- [Auteur](#auteur)

---

##  Description du Projet

Ce projet implémente un pipeline complet de Machine Learning pour la prédiction du risque de crédit bancaire, avec un focus particulier sur la détection et la mitigation du **data drift**. 

### Problématique

Les modèles de crédit sont sensibles aux évolutions démographiques et économiques :
- Vieillissement de la population
- Inflation des salaires
- Hausse des prix immobiliers

Ce projet étudie l'impact de ces changements et propose des stratégies d'adaptation.

---

##  Objectifs

### Phase 1 : Préparation des Données (Q1)
-  Nettoyage du dataset (gestion valeurs manquantes, outliers)
-  Détection et résolution du **data leakage** (variables NB_BAD, NB_LATE)
-  Feature engineering et sélection
-  Gestion du **déséquilibre extrême** (99.7% Good / 0.3% Bad)

### Phase 2 : Évaluation de la Stabilité (Q2)
-  **Cross-Validation stratifiée** (k-fold avec préservation des classes)
-  **Bootstrap** pour estimation de la variance
-  Analyse de la dépendance aux splits (écart-type σ = 0.044)
-  Comparaison des métriques (F1-weighted vs **ROC-AUC**)

### Phase 3 : Optimisation des Hyperparamètres (Q3)
-  **RandomizedSearch** (30 itérations, 8 hyperparamètres)
-  Définition et justification de l'espace de recherche
-  **Validation Curves** pour 2 hyperparamètres majeurs :
  - `max_depth` : identification zones stables/instables
  - `reg_lambda` : analyse du plateau de régularisation
-  Évaluation finale avec **intervalle de confiance à 95%**

### Phase 5 : Simulation et Mesure du Drift (Q5)
-  Création d'un dataset drifté (3 variables modifiées) :
  - `DAYS_BIRTH` : +10 ans (vieillissement démographique)
  - `AMT_INCOME_TOTAL` : +20% (inflation)
  - `AMT_CREDIT` : +15% (hausse prix immobiliers)
-  Calcul de **5 métriques de drift** :
  - Kolmogorov-Smirnov (KS)
  - Wasserstein Distance
  - Population Stability Index (PSI)
  - Jensen-Shannon Divergence (JSD)
  - Chi-Square (variables catégorielles)
-  Analyse de sensibilité des métriques
-  Évaluation de la **dégradation de performance** 

### Phase 6 : Mitigation du Drift (Q6)
-  **Stratégie 1** : Suppression des variables fortement driftées
  - Critère : KS > 0.3 ou PSI > 0.2
  - Résultat : Récupération de 56% de la performance perdue
-  **Stratégie 2** : Réentraînement sur nouveau domaine
  - Protocole sans data leakage (split interne 70/30)
  - Combinaison train original + test drifté
-  **Comparaison des stratégies** (coût, complexité, performance)

---

##  Dataset

- **Source** : Home Credit Default Risk
- **Taille** : ~300,000 clients
- **Classes** : 
  - Good (1) : 99.7% (clients solvables)
  - Bad (0) : 0.3% (clients à risque)
- **Features** : ~25 variables (numériques et catégorielles)
- **Déséquilibre** : Ratio 332:1 → Nécessite métriques adaptées

### Variables Clés
- `DAYS_BIRTH` : Âge du client (en jours, négatif)
- `AMT_INCOME_TOTAL` : Revenu annuel
- `AMT_CREDIT` : Montant du crédit demandé
- `CODE_GENDER` : Genre
- `FLAG_OWN_CAR`, `FLAG_OWN_REALTY` : Possession biens

---

##  Méthodologie

### 1. Prétraitement
```python
# Gestion du déséquilibre
- Métrique principale : ROC-AUC (insensible au déséquilibre)
- StratifiedKFold obligatoire (préservation 99.7/0.3)
- Pas de SMOTE (sur-représentation artificielle)

# Data Leakage
- Suppression de NB_BAD et NB_LATE (leak de la variable target)
- Vérification : corrélation avec TARGET > 0.9
```

### 2. Modélisation
```python
Algorithme : XGBoost (Gradient Boosting)

Hyperparamètres optimaux :
- n_estimators : 235
- max_depth : 5
- learning_rate : 0.1975
- reg_lambda : 4.17 (L2 forte)
- reg_alpha : 0.019 (L1 quasi nulle)
- min_child_weight : 1
- subsample : 0.8
- colsample_bytree : 0.8
```

### 3. Validation
```python
# Cross-Validation
- RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
- 30 scores pour estimation robuste
- IC 95% : [0.703, 0.721]

# Bootstrap
- 30 itérations avec remplacement
- Comparaison avec CV
- Écart-type : 0.017 (vs 0.044 pour CV)
```

### 4. Drift Simulation
```python
# Transformations sur données standardisées
X_test_drift['DAYS_BIRTH'] = X_test['DAYS_BIRTH'] - 1.5σ
X_test_drift['AMT_INCOME_TOTAL'] = X_test['AMT_INCOME_TOTAL'] + 1.0σ
X_test_drift['AMT_CREDIT'] = X_test['AMT_CREDIT'] + 0.75σ

# Métriques calculées
- KS test : max distance entre CDF
- PSI : standard bancaire (seuil = 0.2)
- Wasserstein : sensible aux shifts
```

---

##  Résultats Clés

### Performance Baseline

| Métrique | Score | Interprétation |
|----------|-------|----------------|
| **ROC-AUC** | **0.7322** | Bonne discrimination |
| F1-weighted | 0.997 | Trompeur (déséquilibre) |
| Accuracy | 0.997 | Trompeur (déséquilibre) |

### Optimisation Hyperparamètres (Q3)
```
RandomizedSearch (30 iterations) :
  Score initial : 0.697
  Score optimisé : 0.7298
  Amélioration : +3.3%
```

**Validation Curves :**
- `max_depth` : Zone optimale [4, 5], overfitting à partir de 6
- `reg_lambda` : Plateau stable [3.0, 5.0]

### Impact du Drift (Q5)
```
Performance AVANT drift : 0.7416
Performance APRÈS drift : 0.6158
Dégradation : -0.1258 
```

**Métriques de Drift :**

| Variable | KS | PSI | JSD | Status |
|----------|-----|-----|-----|--------|
| DAYS_BIRTH | 1.00 | 0.82 | 0.34 | **FORT** |
| AMT_INCOME_TOTAL | 1.00 | 0.45 | 0.21 | **FORT** |
| AMT_CREDIT | 1.00 | 0.34 | 0.19 | **FORT** |

### Mitigation du Drift (Q6)

| Stratégie | Score | Amélioration | Taux Récupération |
|-----------|-------|--------------|-------------------|
| Baseline (drift) | 0.6158 | - | - |
| **Suppression variables** | 0.6815 | +0.0657 | **56.4%** |
| Réentraînement | 0.68XX | +0.0XXX | XX% |

**Recommandation :** Stratégie 1 (suppression) pour son rapport coût/efficacité.

---

## 📁 Structure du Projet
```
CreditRisk-ML-Drift-Analysis/
│
├── data/
│   ├── credit_record.csv           # Dataset brut
│   └── application_record.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb   # Q1 - Nettoyage
│   ├── 02_stability_analysis.ipynb # Q2 - CV/Bootstrap
│   ├── 03_hyperparameter_opt.ipynb # Q3 - Optimisation
│   ├── 05_drift_analysis.ipynb     # Q5 - Simulation drift
│   └── 06_drift_mitigation.ipynb   # Q6 - Stratégies
│
├── src/
│   ├── preprocessing.py            # Fonctions nettoyage
│   ├── drift_detection.py          # Métriques drift
│   └── models.py                   # Pipeline ML
│
├── results/
│   ├── validation_curves.png
│   ├── drift_metrics.csv
│   └── comparison_strategies.png
│
├── requirements.txt
└── README.md
```

---

##  Technologies Utilisées

### Langages et Frameworks
- **Python 3.8+**
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Scikit-learn** : Pipeline ML, métriques, validation
- **XGBoost** : Gradient Boosting optimisé
- **SciPy** : Tests statistiques (KS, Chi², Wasserstein)
- **Matplotlib / Seaborn** : Visualisations

### Métriques et Tests
- ROC-AUC, Precision, Recall, F1-Score
- Kolmogorov-Smirnov test
- Population Stability Index (PSI)
- Jensen-Shannon Divergence
- Chi-Square test

---

##  Installation et Exécution

### Prérequis
```bash
Python >= 3.8
pip >= 21.0
```


# Installer dépendances
pip install -r requirements.txt
```

### Fichier requirements.txt
```txt
pandas==1.5.3
numpy==1.24.2
scikit-learn==1.2.2
xgboost==2.0.3
scipy==1.10.1
matplotlib==3.7.1
seaborn==0.12.2
jupyter==1.0.0
```

### Exécution
```bash
# Lancer Jupyter
jupyter notebook

# Exécuter les notebooks dans l'ordre :
# 01 → 02 → 03 → 05 → 06

###  Pièges Évités

- ❌ Data leakage (NB_BAD, NB_LATE)
- ❌ Overfitting sur hyperparams (validation curves)
- 
###  Concepts Avancés Appliqués

- Biais-variance tradeoff
- Learning curves analysis
- Domain adaptation
- Distribution shift (covariate shift)
- Regularization (L1 vs L2)
- Statistical hypothesis testing

---

##  Enseignements

### Ce qui Fonctionne

 **XGBoost avec forte régularisation L2**
- Robuste au déséquilibre
- Gère bien les interactions

 **Suppression variables driftées**
- Simple à implémenter
- Récupération significative (56%)

 **Monitoring continu du drift**
- PSI calculé périodiquement
- Alerte si PSI > 0.2




##  Pour Aller Plus Loin

### Améliorations Possibles

1. **Feature Engineering Avancé**
   - Ratios (Dette/Revenu, Crédit/Patrimoine)
   - Variables temporelles (ancienneté emploi)
   - Agrégations (moyenne revenus par région)

2. **Ensemble Methods**
   - Stacking (XGBoost + LightGBM + CatBoost)
   - Blending avec pondération

3. **Drift Adaptation Automatique**
   - Détection temps réel (monitoring)
   - Réentraînement déclenché si PSI > seuil
   - A/B testing nouvelles versions

4. **Explainability**
   - SHAP values par client
   - Analyse contrefactuelle
   - Rapport réglementaire (RGPD)



## Références

1. **Gama, J. et al. (2014)** - "A Survey on Concept Drift Adaptation"
2. **Rabanser, S. et al. (2019)** - "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift"
3. **Lu, J. et al. (2018)** - "Learning under Concept Drift: A Review"
4. **Hastie, T. et al. (2009)** - "The Elements of Statistical Learning"
5. **Chen, T. & Guestrin, C. (2016)** - "XGBoost: A Scalable Tree Boosting System"

---
