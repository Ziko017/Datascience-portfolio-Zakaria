import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# Charger le fichier CSV
file_path = 'Dataset_Thuy (1).csv'
df = pd.read_csv(file_path)

# Définir la variable cible
y = df["Infringment"].astype(int)

# Supprimer la colonne cible et les colonnes non-informatives
X = df.drop(columns=["Infringment", "pub_nbr"])

# Imputation des valeurs manquantes (médiane)
X.fillna(X.median(numeric_only=True), inplace=True)

# Encodage fréquentiel des colonnes catégorielles
print("Column data types before encoding:")
print(X.dtypes)

for col in X.select_dtypes(include=['object', 'category']).columns:
    freq = X[col].value_counts(normalize=True)
    X[col] = X[col].map(freq)

X = X.apply(pd.to_numeric, errors='coerce')

print("\nFirst few rows after encoding and numeric conversion:")
print(X.head())

print("\nNumber of NaN values per column:")
print(X.isna().sum())

# Séparer les données AVANT SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Appliquer SMOTE uniquement sur l'ensemble d'entraînement
smote = SMOTE(random_state=42, sampling_strategy='minority')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Modèle Elastic Net
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=5000, random_state=42)
elastic_net_model.fit(X_train_scaled, y_train_resampled)

# Prédictions continues et binarisation (seuil 0.5)
y_train_pred_cont = elastic_net_model.predict(X_train_scaled)
y_test_pred_cont = elastic_net_model.predict(X_test_scaled)
y_train_pred = (y_train_pred_cont >= 0.5).astype(int)
y_test_pred = (y_test_pred_cont >= 0.5).astype(int)

# Évaluation
print("### Résultats du modèle Elastic Net (seuil 0.5) ###")
print(f"Accuracy (train): {accuracy_score(y_train_resampled, y_train_pred):.4f}")
print(f"Accuracy (test): {accuracy_score(y_test, y_test_pred):.4f}")
print("Matrice de confusion (test):\n", confusion_matrix(y_test, y_test_pred))
print("Rapport de classification (test):\n", classification_report(y_test, y_test_pred, digits=4))

# Sélection de variables importantes
selector = SelectFromModel(elastic_net_model, threshold="1.5*mean", prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print(f"\nNombre de features sélectionnées (seuil strict) : {len(selected_features)}")
print(f"Features importantes : {list(selected_features)}")

# Réentraînement du modèle Elastic Net sur les features sélectionnées
final_model = ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=5000, random_state=42)
final_model.fit(X_train_selected, y_train_resampled)

# Prédictions finales
y_test_pred_final_cont = final_model.predict(X_test_selected)
y_test_pred_final = (y_test_pred_final_cont >= 0.5).astype(int)

print("\n### Résultats après sélection stricte avec Elastic Net ###")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_final):.4f}")
print("Matrice de confusion:\n", confusion_matrix(y_test, y_test_pred_final))
print("Rapport de classification:\n", classification_report(y_test, y_test_pred_final, digits=4))
print("\n### Résultats après sélection stricte avec Elastic Net ###")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_final):.4f}")
print("Matrice de confusion:\n", confusion_matrix(y_test, y_test_pred_final))
print("Rapport de classification:\n", classification_report(y_test, y_test_pred_final, digits=4))
