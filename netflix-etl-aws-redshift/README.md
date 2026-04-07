# ETL Pipeline & Data Warehouse sur AWS – Netflix (Contribution personnelle)

Dans ce projet, je me suis occupé de la mise en place de l’architecture ETL et Data Warehouse sur AWS afin de structurer les données et faciliter leur exploitation par mes collègues dans Power BI.

J’ai conçu un pipeline sécurisé permettant de centraliser les données, créer les schémas analytiques dans Amazon Redshift et synchroniser directement les datasets avec l’outil de visualisation via un accès réseau contrôlé dans un VPC.

---

## Architecture AWS

### Flux de données

S3 → IAM Role → Redshift (Data Warehouse) → VPC Endpoint → Power BI  

Services complémentaires : CloudWatch, CloudTrail, Security Groups, SNS/SQS

---

## Stockage des données – Amazon S3

Les datasets sources sont stockés dans Amazon S3 avec une configuration optimisée :

- storage class : Standard – Infrequent Access (Standard-IA)
- upload via Multipart Upload
- chiffrement SSE-S3
- protection des objets avec S3 Object Lock

Les fichiers sont transmis au Data Warehouse via S3 Pre-signed URL afin de permettre un accès temporaire sécurisé lors du chargement dans Redshift.

---

## Gestion des accès – IAM Policy

Configuration d’un IAM Role permettant à Redshift d’accéder aux objets S3 :

- Managed Policy ReadOnly
- accès sécurisé aux fichiers via Pre-signed URL
- principe du least privilege
- authentification sécurisée entre services AWS

---

## Data Warehouse – Amazon Redshift

Création d’un environnement analytique optimisé :

- création des schémas analytiques
- structuration des tables de faits et dimensions
- optimisation pour requêtes OLAP
- stockage columnar
- architecture distribuée (leader node + compute nodes)

Objectif :
faciliter l’analyse des données et améliorer la performance des visualisations Power BI.

---

## Transformations ETL

Les données nécessitaient peu de nettoyage car les datasets étaient déjà relativement structurés.

Transformations principales réalisées :

- nettoyage léger des colonnes (format texte, valeurs manquantes)
- normalisation des titres de contenus
- jointure des datasets sur le couple :
  
  show_title + director
  
- création d’une feature d’ingénierie basée sur la durée du contenu :

  - short : durée ≤ 30 min
  - standard : durée ≤ 60 min
  - long : durée > 60 min

Ces transformations ont permis d’uniformiser les sources de données et de faciliter l’analyse dans Power BI.

---

## Création des schémas analytiques (SQL)

Des requêtes SQL utilisées dans Amazon Redshift pour structurer le Data Warehouse, schéma en flocon (voir le pdf de la présentation)


## Réseau sécurisé – VPC

Déploiement du cluster dans un VPC :

- Allow IPv4 public access activé
- port 5439 configuré pour Redshift
- endpoint sécurisé
- isolation réseau

---

## Security Groups

Configuration des règles réseau :

- ouverture du port 5439
- restriction des IP autorisées
- contrôle du trafic

---

## Monitoring et audit

CloudWatch :
- monitoring des performances
- logs

CloudTrail :
- audit des actions
- traçabilité

---

## Intégration Power BI

Connexion directe au Data Warehouse :

- accès aux datasets structurés
- création simplifiée de dashboards
- requêtes optimisées

---

## Ma contribution

J’ai conçu et implémenté cette architecture AWS afin de :

- structurer les données dans Redshift
- sécuriser les flux via IAM et VPC
- permettre une connexion directe avec Power BI
- faciliter le travail de data visualisation pour mes collègues

---

## Compétences démontrées

ETL Pipeline  
AWS Cloud  
Amazon S3  
IAM Roles & Policies  
Amazon Redshift  
VPC  
Data Warehouse  
Power BI
