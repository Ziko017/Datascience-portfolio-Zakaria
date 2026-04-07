ETL Pipeline & Data Warehouse sur AWS – Netflix
Contribution personnelle

Dans ce projet, je me suis occupé de la mise en place de l’architecture ETL et Data Warehouse sur AWS afin de structurer les données et faciliter leur exploitation par mes collègues dans Power BI.

J’ai conçu un pipeline sécurisé permettant de centraliser les données, créer les schémas analytiques dans Amazon Redshift et synchroniser directement les datasets avec l’outil de visualisation via un accès réseau contrôlé dans un VPC.

Architecture AWS

Flux de données :

S3 → IAM Role → Redshift (Data Warehouse) → VPC Endpoint → Power BI

Services complémentaires : CloudWatch, CloudTrail, Security Groups, SNS/SQS

Stockage des données – Amazon S3

Les datasets sources sont stockés dans Amazon S3 dans une classe optimisée pour le coût et l’accès analytique :

Storage class : Standard – Infrequent Access (Standard-IA)
Upload optimisé via Multipart Upload
Chiffrement des données avec SSE-S3
Protection de l’intégrité des objets avec S3 Object Lock

Les fichiers sont ensuite transmis au Data Warehouse via S3 Pre-signed URL afin de permettre un accès sécurisé temporaire pour le chargement dans Redshift.

Gestion des accès – IAM Policy

Un IAM Role a été configuré pour permettre au cluster Redshift d’accéder aux données stockées dans S3.

Configuration :

création d’une Managed Policy ReadOnly
accès restreint aux ressources nécessaires
autorisation d’accès aux objets S3 via URL sécurisée
application du principe du moindre privilège

Ce mécanisme permet d’éviter l’exposition de credentials tout en garantissant une communication sécurisée entre services AWS.

Data Warehouse – Amazon Redshift

Le Data Warehouse centralise les données transformées afin de faciliter l’analyse dans Power BI.

Mise en place :

création des schémas analytiques
structuration des tables pour requêtes BI
optimisation pour charges OLAP
organisation des données pour simplifier la création de visualisations

Redshift permet :

stockage columnar performant
scalabilité horizontale
intégration native avec l’écosystème AWS
Réseau sécurisé – VPC

Le cluster Redshift est déployé dans un Virtual Private Cloud (VPC) afin de sécuriser les flux de données.

Configuration réseau :

VPC public avec Allow IPv4 public access activé
ouverture du port 5439 pour la connexion Redshift
configuration d’un endpoint partagé pour l’accès depuis Power BI
isolation du Data Warehouse dans un environnement contrôlé
Security Groups

Les Security Groups contrôlent l’accès au cluster :

autorisation du trafic entrant sur le port 5439
filtrage des IP autorisées
sécurisation de la connexion Power BI → Redshift
Monitoring et audit

CloudWatch :

monitoring des performances
logs système
suivi des requêtes

CloudTrail :

historique des accès
audit des actions sur l’infrastructure

SNS / SQS :

gestion des notifications entre services
Pipeline ETL

Extract

récupération datasets depuis Kaggle
stockage dans S3

Transform

nettoyage avec Python (Pandas)
normalisation des colonnes
jointure des datasets
création de variables analytiques

Load

chargement dans Redshift
création des schémas analytiques
structuration des tables pour Power BI
Impact pour l’équipe

Cette architecture permet :

accès direct aux données depuis Power BI
meilleure performance des requêtes analytiques
datasets structurés pour la visualisation
environnement sécurisé et scalable
simplification du travail des analystes
Stack technique

AWS

S3
IAM
Redshift
VPC
CloudWatch
CloudTrail
SNS
SQS

Data

Python
Pandas
SQL

BI

Power BI
