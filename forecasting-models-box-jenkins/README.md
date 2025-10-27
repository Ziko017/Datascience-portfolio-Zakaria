# Prévision temporelle — Méthode Box–Jenkins (R)

**Objectif :** appliquer la méthodologie Box–Jenkins sur les ventes mensuelles du commerce de détail au Royaume-Uni (2017–2023).  

**Méthodes :**  
- Analyse de la stationnarité (test KPSS, ACF, PACF)  
- Modélisation ARIMA et SARIMA avec sélection par AIC/BIC  
- Validation des résidus (test de Ljung–Box)  
- Exploration de modèles hybrides **SARIMA–XGBoost** et **SARIMA–SVR** pour la composante non linéaire  

**Résultats :**  
Le modèle **SARIMA(3,1,1)(1,1,1)[12]** a offert les meilleures performances, reproduisant efficacement la tendance et la saisonnalité annuelle des ventes.  
Les modèles hybrides ont été évalués (RMSE, MAE, MAPE), sans gain significatif car les résidus étaient déjà proches d’un bruit blanc.  

**Compétences :** R · Time Series Forecasting · SARIMA · XGBoost · SVR · Model Evaluation  

 Rapport complet : `ForecastingModels.pdf`
