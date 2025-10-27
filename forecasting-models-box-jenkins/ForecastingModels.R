setwd("H://Documents//ING3//forecasting models//TP R")


library(tseries)
library(urca)
library(fpp2)
library(urca)
library(forecast)
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#Partie 1 

#Q1-Simuler un processus ARIMA(p,d,q) stationnaire (d=1, p≥2, q≥1) avec une fonction 
#personnalisée Arima_sim.

#Entrées : 
#p, d, q : ordres du modèle
#phi[1:p] : coefficients AR
#theta[1:q] : coefficients MA
#sigma2 : variance du bruit blanc
#n : longueur souhaitée de la série

arima_sim <- function(p, d, q, phi = NULL, theta = NULL, sigma2 = 1, n = 100, transitoire = 70) {
 
  # Étape 1 : Générer le bruit blanc ε_t
 
  e <- rnorm(n + transitoire + q, mean = 0, sd = sqrt(sigma2))
  
  # Étape 2 : Initialiser la série y_t
 
  y <- numeric(n + transitoire + max(p, q))
  
  
  # -------------------------------------
  # Étape 3 : Générer la partie ARMA(p, q)

  for (t in (max(p, q) + 1):(n + transitoire)) {
    # Partie AR : somme des valeurs passées de y
    ar_part <- 0
    if (p > 0) {
      for (i in 1:p) {
        ar_part <- ar_part + phi[i] * y[t - i]
      }
    }
    
    # Partie MA : somme des erreurs passées
    ma_part <- 0
    if (q > 0) {
      for (j in 1:q) {
        ma_part <- ma_part + theta[j] * e[t - j]
      }
    }
    
    # Valeur simulée
    y[t] <- ar_part + ma_part + e[t]
  }
  
  # Retirer la période de stabilisation
  y <- y[(transitoire + 1):(transitoire + n)]
  
  # -----------------------------------------------------
  # Étape 4 : Intégration si d > 0 (pour ARIMA)
  
  if (d > 0) {
    for (i in 1:d) {
      y <- cumsum(y)   # cumul successif pour introduire la différenciation inverse
    }
  }
  
  return(y)
}
# example d'application

set.seed(123)

# Série simulée avec notre fonction
y <- arima_sim(
  p = 3, d = 1, q = 2,
  phi = c(1, -0.3, 0.2),
  theta = c(0.4, -0.2),
  sigma2 = 1,
  n = 200
)

# Série simulée avec arima.sim de R
serie <- arima.sim(
  model = list(order = c(3,1,2), ar = c(1, -0.3, 0.2), ma = c(0.4, -0.2)),
  n = 200
)

# Tracer les deux séries sur le même graphique
plot(y, type = "l", col = "blue", lwd = 2,
     ylim = range(c(y, serie)),
     main = "Comparaison ARIMA(3,1,2)",
     ylab = "Valeur", xlab = "Temps")
lines(serie, col = "red", lwd = 2)

# Ajouter une légende
legend("topleft", legend = c("arima_sim", "arima.sim"), 
       col = c("blue", "red"), lwd = 2)

# On remarque que les deux séries suivent presque les mêmes variations, mais elles diffèrent naturellement sur certaines valeurs en raison de l'aléa.






#Q2-

Forecast_Per <- function(model, h = 5) {
  #----------------------------------------------------------
  # Prévision pas à pas pour un modèle SARIMA(p,1,q)
  # avec développement (1 - B)*phi(B)
  #----------------------------------------------------------
  
  # Extraction des composantes du modèle
  phi   <- model$model$phi        # Coefficients AR
  theta <- model$model$theta      # Coefficients MA
  d     <- model$arma[6]          # Ordre de différenciation (ici, d = 1)
  p     <- model$arma[1]
  q     <- model$arma[2]
  
  # Série ajustée et résidus
  y <- as.numeric(model$x)              # Série observée
  e <- as.numeric(residuals(model))     # Résidus
  
  # Initialisation du vecteur des prévisions finales
  forecasts <- numeric(h)
  
  #----------------------------------------------------------
  # Calcul du polynôme élargi : (1 - B) * phi(B)
  #----------------------------------------------------------
  phi_expanded <- numeric(p + 1)
  phi0 <- 1  # correspond à 1 du polynôme (1 - B)
  
  for (i in 0:p) {
    if (i == 0) {
      phi_expanded[i + 1] <- (phi0 + (ifelse(p >= 1, phi[1], 0)))
    } else if (i < p) {
      phi_expanded[i + 1] <- phi[i+1] - phi[i]
    } else {
      phi_expanded[i + 1] <-  - phi[p]
    }
  }
  
  #----------------------------------------------------------
  # Boucle de prévision pas à pas
  #----------------------------------------------------------
  for (step in 1:h) {
    y_pred <- 0
    
    # --- Partie AR (avec phi_expanded) ---
    for (i in 1:length(phi_expanded)) {
      if (length(y) - i >= 0) {
        y_pred <- y_pred + phi_expanded[i] * y[length(y) - i+1 ]
        
      }
      
    }
    
    # --- Partie MA ---
    if (q > 0) {
      for (j in 1:q) {
        if (length(e) - j >= 0) {
          y_pred <- y_pred + theta[j] * e[length(e) - j +1]
        }
      }
    }
    
    # Stocker la prévision
    forecasts[step] <- y_pred
    y <- c(y, y_pred)
    e <- c(e, 0)
    
    
    
    
    
  }
  
  return(forecasts)
  
}


#Deuxième approche : 
Forecast_Per_2 <- function(model, h = 5) {
  
  # Extraction des composantes
  phi   <- model$model$phi        # Coefficients AR
  theta <- model$model$theta      # Coefficients MA
  d     <- model$arma[6]          # Ordre de différenciation (ici, d = 1)
  p     <- model$arma[1]
  q     <- model$arma[2]
  
  # Série ajustée et résidus
  y <- as.numeric(model$x)
  e <- as.numeric(residuals(model))
  
  forecasts <- numeric(h)
  
  # Boucle pas-à-pas
  for (step in 1:h) {
    ar_part <- if (p > 0) sum(phi * rev(tail(y, p))) else 0
    ma_part <- if (q > 0) sum(theta * rev(tail(e, q))) else 0
    
    y_pred <- ar_part + ma_part + 0  # erreurs futures supposées nulles
    forecasts[step] <- y_pred
    
    # Mise à jour de y et e pour la prochaine étape
    y <- c(y, y_pred)
    e <- c(e, 0)
  }
  
  # Reconstruction si d > 0
  if (d > 0) {
    forecasts <- cumsum(forecasts)
  }
  
  return(forecasts)
}



#Comparaison simple : 


serie <- arima.sim(
  model = list(order = c(3,1,2), ar = c(1, -0.3, 0.2), ma = c(0.4, -0.2)),
  n = 100
)
model <- Arima(serie, order = c(3,1,2))
model$model$phi

manual_forecast= Forecast_Per(model, h=2)


manual_forecast_2 = Forecast_Per_2(model,h=2)
auto_forecast <- forecast(model, h = 2)
print(manual_forecast)
print(auto_forecast)
print(manual_forecast_2)



#Q3- comparaison MSE 

set.seed(123)

phi <- c(0.5, -0.3, 0.2, 0.1, -0.05, 0.04, -0.02)   # 7 coefficients AR
theta <- c(0.4, -0.3, 0.2, 0.1, -0.05, 0.03, 0.02, -0.01)# 8 coefficients MA
p=7
q=8

phi_1= c(0.5, -0.3, 0.2)
theta_1= c(0.4, -0.3, 0.2)
p_1 =3
q_1=3
serie <- arima.sim(
  model = list(order = c(p_1,1,q_1), ar = phi_1, ma = theta_1),
  n = 100
)


# Séparation en train et test
train <- serie[1:70]
test <- serie[71:100]
h= 20
#h <- length(test)

# Ajustement du modèle sur l'échantillon d'entraînement
fit <- arima(train, order = c(p_1,1,q_1))

# ------------------------------
# Prévision avec forecast()
# ------------------------------
fc_forecast <- forecast(fit, h = h)
pred_forecast <- fc_forecast$mean

# ------------------------------
# Prévision avec predict()
# ------------------------------
fc_predict <- predict(fit, n.ahead = h)
pred_predict <- fc_predict$pred

# ------------------------------
# Prévision avec ta fonction Forecast_Per()
# ------------------------------
pred_custom <- Forecast_Per(fit, h = h)


# ------------------------------
# Prévision avec ta fonction Forecast_Per_2()

# ------------------------------

pred_custom_2 <- Forecast_Per_2(fit, h = h)

# ------------------------------
# Calcul des MSE pour chaque méthode
# ------------------------------
mse <- function(actual, predicted) mean((actual - predicted)^2)

MSE_forecast <- mse(test[1:h], pred_forecast)
MSE_predict <- mse(test[1:h], pred_predict)
MSE_custom <- mse(test[1:h], pred_custom)
MSE_custom_2 = mse(test[1:h], pred_custom_2)

mse_table <- data.frame(
  Méthode = c("forecast()", "predict()", "Forecast_Per()", "Forecast_Per_2()"),
  MSE = c(MSE_forecast, MSE_predict, MSE_custom, MSE_custom_2 )
)
print(mse_table)

# ------------------------------
#  Visualisation comparative
# ------------------------------

# Calculer les limites automatiquement en fonction des trois séries
y_min <- min(c(test[1:h], pred_forecast, pred_custom, pred_custom_2))
y_max <- max(c(test[1:h], pred_forecast, pred_custom, pred_custom_2))

# Ajouter un petit “padding” de 5% pour que ça respire
y_range <- y_max - y_min
ylim_range <- c(y_min - 0.05*y_range, y_max + 0.05*y_range)

# Plot
plot(1:h, test[1:h], type = "l", lwd = 2, col = "black",
     main = "Comparaison des prévisions (ARIMA(3,1,3))",
     ylab = "Valeur", xlab = "Temps (test set)",
     ylim = ylim_range)

# Ajouter les prévisions
lines(1:h, pred_forecast, col = "blue", lwd = 2)
lines(1:h, pred_custom, col = "red", lwd = 2, lty = 2)
lines (1:h, pred_predict, col = "green", lwd = 2, lty = 2)
lines(1:h, pred_custom_2, col = "pink", lwd = 2, lty = 2)

# Légende
legend("topleft", legend = c("Test set", "forecast()", "forcast_pred" , "Forecast_Per()","Forecast_Per_2()" ),
       col = c("black", "blue","green", "red", "pink"), lty = c(1,1,2), lwd = 2)


#---------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
#Partie : 2 


##==============Question1=============##
data <- Copie_de_DataSerie_Temporelle
y <- ts(data[[2]], frequency = 12)
plot(y, main = "Ventes mensuelles (2017–2023)", col = "darkblue", ylab = "Montant", xlab = "Temps")

par(mfrow=c(1,2))
acf(y)      
pacf(y)
par(mfrow=c(1,1))

#Vérification de la variance (Box–Cox)
lambda <- BoxCox.lambda(y)
cat("Lambda estimé =", lambda, "\n")
# Lambda estimé = 0.848407
# On déduit donc que notre variance est déja stable

#===============Question2==============#
Test <- ur.kpss(y)
summary(Test)# KPSS (H0 : stationnaire)
#Value of test-statistic is: 1.282 > à toutes les valeurs critiques
#Série non stationnaire

#Méthode 1: différenciation manuelle
y_diff1 <- diff(y, differences = 1)
y_diff2 <- diff(y_diff1, lag = frequency(y), differences = 1)
y_stat_P <- y_diff2 

Test2 <- ur.kpss(y_stat_P)
summary(Test2)
##Notre série est devenu stationnaire vérifié grâce au Test
#Value of test-statistic is: 0.0215 < à toutes les valeurs critiques
#Du coup on accepte l'hyptohèse H0 stationnarité
#du coup vu qu'on fait une seule différenciation 
#On déduit que d=1 et D=1


#Méthode 2 pour obtenir directement le nombre de différenciation fait
d <- ndiffs(y)                    # différences non saisonnières
D <- nsdiffs(y, m = frequency(y)) # différences saisonnières (données mensuelles)
cat("d =", d, " | D =", D, "\n")

y_stat <- y
if (D > 0) y_stat <- diff(y_stat, lag = frequency(y), differences = D)
if (d > 0) y_stat <- diff(y_stat, differences = d)

Test3 <- ur.kpss(y_stat)
summary(Test3)
#Notre série est devenu stationnaire vérifié grâce au Test
#Value of test-statistic is: 0.0215 < à toutes les valeurs critiques
#Du coup on accepte l'hyptohèse H0 stationnarité

par(mfrow=c(1,2))
acf(y_stat)      
pacf(y_stat)
par(mfrow=c(1,1))

##===========Question3============##
#Effectuer le split train/test
n <- length(12)
test_len <- 12              # 12 mois = 1 an de test
train <- head(y, n - test_len) # On train sur les premières 70 observations
test  <- tail(y, test_len) # On teste sur les 12 dernière observation

#split sur y_stat (pour ARMA)
n_s <- length(y_stat)
train_stat <- head(y_stat, n_s - test_len)
test_stat  <- tail(y_stat, test_len)

###===Ajustons nos modèles ARMA/ARIMA/SARIMA===###
#Ajustons notre modèle ARMA
for(p in 0:3){
  for(q in 0:3){
    fit <- tryCatch(
      Arima(na.omit(train_stat), order=c(p,0,q), seasonal=FALSE),
      error = function(e) NULL
    )
    if(!is.null(fit)){
      cat("ARMA(",p,",0,",q,") : AIC =", AIC(fit), "\n")
    }
  }
}

#Ajustons notre modèle ARIMA
for(p in 0:8){
  for(q in 0:3){
    fit <- tryCatch(
      Arima(na.omit(train), order=c(p,1,q), seasonal=FALSE),
      error = function(e) NULL
    )
    if(!is.null(fit)){
      cat("ARIMA(",p,",1,",q,") : AIC =", AIC(fit), "\n")
    }
  }
}
#Notre série est saisonnière du coup on inclut P et Q
#Ajustons notre modèle SARIMA
#On va pas aller trop loin pour éviter le surajustement

for(p in 0:3){
  for(q in 0:3){ 
    for(P in 0:1){
      for(Q in 0:1){
        fit <- Arima(train, order=c(p,1,q), seasonal=list(order=c(P,1,Q), period=12))
        cat("SARIMA(",p,",1,",q,")(",P,",1,",Q,")[12] : AIC =", AIC(fit), "\n") } 
    } 
  } 
}
# top 3 des modèles sont:
# Après exploration des combinaisons (p,q,P,Q) dans les plages :
#   p = 0:3 , q = 0:3 , P = 0:1 , Q = 0:1
# les trois modèles ayant obtenu les plus faibles valeurs de AIC 
# 1- SARIMA(3,1,0)(0,1,1)[12]  →  AIC = 1022.788 |
# 2- SARIMA(3,1,1)(1,1,1)[12]  →  AIC = 1024.650 
# 3- SARIMA(3,2,0)(0,1,1)[12]  →  AIC ≈ 1027–1030 

# Ces modèles présentent :
# des critères AIC nettement inférieurs aux autres combinaisons testées,
# ==========Modèle 1: SARIMA(3,1,0)(0,1,1)[12]
fit1 <- Arima(y, order = c(3,1,0),
              seasonal = list(order = c(0,1,1), period = 12))
summary(fit1)
autoplot(forecast(fit1, h = 12)) + 
  ggtitle("Prévisions - SARIMA(3,1,0)(0,1,1)[12]")
checkresiduals(fit1)

# ======== Modèle 2 : SARIMA(3,1,1)(1,1,1)[12] ========
fit2 <- Arima(y, order = c(3,1,1),
              seasonal = list(order = c(1,1,1), period = 12))
summary(fit2)
autoplot(forecast(fit2, h = 12)) + 
  ggtitle("Prévisions - SARIMA(3,1,1)(1,1,1)[12]")
checkresiduals(fit2)

# ======== Modèle 3 : SARIMA(3,2,0)(0,1,1)[12] ========
fit3 <- Arima(y, order = c(3,2,0),
              seasonal = list(order = c(0,1,1), period = 12))
summary(fit3)
autoplot(forecast(fit3, h = 12)) + 
  ggtitle("Prévisions - SARIMA(3,2,0)(0,1,1)[12]")
checkresiduals(fit3)

# 1.SARIMA(3,1,0)(0,1,1)[12] : p-value = 0.0309 < 0.05 → résidus autocorrélés 
# 2.SARIMA(3,1,1)(1,1,1)[12] : p-value = 0.5054 > 0.05 → résidus aléatoires 
# 3.SARIMA(3,2,0)(0,1,1)[12] : p-value = 0.0332 < 0.05 → résidus autocorrélés 
#
# Conclusion : le modèle SARIMA(3,1,1)(1,1,1)[12] est retenu.
# Il offre un compromis optimal entre AIC faible et résidus non autocorrélés.

#On a testé avec auto.arima() pour comparer
fit_arma  <- auto.arima(y_stat, seasonal = FALSE)
fit_arima <- auto.arima(train, seasonal = FALSE)
fit_sarima <- auto.arima(train, seasonal = TRUE)

# Résumés
cat("\n--- ARMA sur y_stat ---\n");   print(fit_arma)
cat("\n--- ARIMA sur y ---\n");      print(fit_arima)
cat("\n--- SARIMA sur y ---\n");     print(fit_sarima)

#Meilleur modèle retenue
checkresiduals(fit_sarima)
#On a trouvé deux modèles différents mais qui sont satisfaisants:
#le modèle SARIMA(3,1,1)(1,1,1)[12] est retenu via des recherches sur des plages de valeurs
#le modèle SARIMA(3,0,0)(1,1,0)[12] est retenu par auto.arima().


#On trouve que auto.arima() nous offre pas le modèle le plus pertinent
#Du coup le modele qu'on garde est bien SARIMA(3,1,1)(1,1,1)
fit_final <- Arima(y, order = c(3,1,1),
                   seasonal = list(order = c(1,1,1), period = 12))

k <- 12  # 12 mois à prédire
forecast_valeurs <- forecast(fit_final, h = k)
print(forecast_valeurs)

autoplot(forecast_valeurs) +
  ggtitle("Prévision sur 12 mois - SARIMA(3,1,1)(1,1,1)[12]") +
  xlab("Temps") + ylab("Ventes mensuelles") +
  theme_minimal()
#Le modèle ne sous- ni sur-prédit pas visiblement, ce qui confirme qu’il a bien capturé la structure saisonnière.

res <- residuals(fit_final)

# Résidus dans le temps
autoplot(res) +
  ggtitle("Résidus du modèle SARIMA(3,1,1)(1,1,1)[12]") +
  xlab("Temps") + ylab("Résidu") +
  theme_minimal()
#prévisions un peu plus incertaines lors de chocs

# Histogramme + densité
ggplot(data.frame(res), aes(x = res)) +
  geom_histogram(aes(y = ..density..), bins = 15, fill = "lightblue", color = "black") +
  geom_density(color = "red", size = 1) +
  ggtitle("Distribution des résidus") +
  theme_minimal()

par(mfrow=c(1,2))
acf(res, main="ACF des résidus")
pacf(res, main="PACF des résidus")
par(mfrow=c(1,1))
#Il s'agit bien d'un bruit blanc

#--------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#Partie 4: 

##########======================PARTIE4 (SARIMA + XGB + SVR)===========================#########
##########======================PARTIE4 (SARIMA + XGB + SVR + RNN)===========================#########
library(xgboost)
library(e1071)
library(ggplot2)
library(keras)
library(tensorflow)

# Données & split 
stopifnot(exists("y"))
m         <- frequency(y)      # 12 pour mensuel
H         <- 12                # horizon de test
n         <- length(y)
train2    <- head(y, n - H)
test2     <- tail(y, H)

# Prévisions SARIMA
fc_sarima  <- forecast(fit_final, h = H)
yhat_sarima_test <- as.numeric(fc_sarima$mean)

# ==== 2) Modèles sur les résidus du train ====
res_tr <- as.numeric(residuals(fit_final))

# Fonction pour créer les lags
make_lags <- function(x, L){
  E <- embed(x, L + 1)
  y <- E[,1]
  X <- E[,-1, drop = FALSE]
  list(y = y, X = X)
}
L <- 12
lagged <- make_lags(res_tr, L)

# ----- a) XGBoost -----
dtrain <- xgb.DMatrix(data = as.matrix(lagged$X), label = lagged$y)
params <- list(
  objective = "reg:squarederror",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  nthread = 2
)
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data   = dtrain,
  nrounds = 1000,
  watchlist = list(train = dtrain),
  verbose = 0,
  early_stopping_rounds = 50
)

predict_residuals_xgb <- function(model, last_res, h, L){
  out <- numeric(h)
  buf <- as.numeric(tail(last_res, L))
  for(i in 1:h){
    Xnew <- matrix(rev(buf), nrow = 1)
    out[i] <- as.numeric(predict(model, xgb.DMatrix(Xnew)))
    buf <- c(buf[-1], out[i])
  }
  out
}

res_pred_xgb <- predict_residuals_xgb(xgb_model, last_res = tail(res_tr, L), h = H, L = L)

# ----- b) SVR -----
svr_model <- svm(lagged$X, lagged$y, type = "eps-regression", kernel = "radial", cost = 100, gamma = 0.1, epsilon = 0.01)

predict_residuals_svr <- function(model, last_res, h, L){
  out <- numeric(h)
  buf <- as.numeric(tail(last_res, L))
  for(i in 1:h){
    Xnew <- matrix(rev(buf), nrow = 1)
    out[i] <- as.numeric(predict(model, Xnew))
    buf <- c(buf[-1], out[i])
  }
  out
}

res_pred_svr <- predict_residuals_svr(svr_model, last_res = tail(res_tr, L), h = H, L = L)

# ==== 3) Prévisions hybrides ====
yhat_hybrid_xgb <- yhat_sarima_test + res_pred_xgb
yhat_hybrid_svr <- yhat_sarima_test + res_pred_svr

# ==== 4) Métriques ====
RMSE <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))
MAE  <- function(y, yhat) mean(abs(y - yhat), na.rm = TRUE)
MAPE <- function(y, yhat) mean(abs((y - yhat)/y), na.rm = TRUE) * 100
sMAPE <- function(y, yhat) mean(200 * abs(y - yhat) / (abs(y) + abs(yhat)), na.rm = TRUE)
MASE <- function(y_train, y_test, yhat, m){
  scale <- mean(abs(diff(y_train, lag = m)), na.rm = TRUE)
  mean(abs(y_test - yhat), na.rm = TRUE) / scale
}

metrics <- function(y_tr, y_te, yhat, m){
  c(
    RMSE  = RMSE(y_te, yhat),
    MAE   = MAE(y_te, yhat),
    MAPE  = MAPE(y_te, yhat),
    sMAPE = sMAPE(y_te, yhat),
    MASE  = MASE(y_tr, y_te, yhat, m)
  )
}

met_sarima <- metrics(train2, test2, yhat_sarima_test, m)
met_hybrid_xgb <- metrics(train2, test2, yhat_hybrid_xgb, m)
met_hybrid_svr <- metrics(train2, test2, yhat_hybrid_svr, m)

cat("\n=== Métriques (Test, h=12) ===\n")
print(rbind(
  SARIMA = met_sarima,
  HYBRID_SARIMA_XGB = met_hybrid_xgb,
  HYBRID_SARIMA_SVR = met_hybrid_svr
), digits = 4)

# ==== 5) Plots comparatifs ====
df_plot <- data.frame(
  date = time(tail(y, H)),
  Observé = as.numeric(test2),
  SARIMA  = yhat_sarima_test,
  HYBRID_XGB = yhat_hybrid_xgb,
  HYBRID_SVR = yhat_hybrid_svr
)

ggplot(df_plot, aes(x = date)) +
  geom_line(aes(y = Observé)) +
  geom_line(aes(y = SARIMA), linetype = 2) +
  geom_line(aes(y = HYBRID_XGB), linetype = 1, color = "blue") +
  geom_line(aes(y = HYBRID_SVR), linetype = 1, color = "red") +
  labs(title = "Test: Observé vs Prévu (SARIMA vs Hybrides XGB/SVR)",
       x = "Temps", y = "Valeur") +
  theme_minimal()
