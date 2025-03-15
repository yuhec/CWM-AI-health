# CWM-AI-health

# 🚀 Démystifions l'IA : Détection du Cancer du Sein avec TensorFlow 🚀

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/drive/folders/1qIM_ixEGTFX5vL4ImK6Bk26dRNqfsV0E?usp=sharing)

## Bienvenue !

Cet atelier interactif vous guide à travers les étapes de création d'un modèle d'intelligence artificielle pour la détection du cancer du sein à partir d'images d'échographie. Aucun prérequis technique n'est nécessaire, juste de la curiosité et l'envie d'apprendre !

## Objectifs

- Comprendre les principes de base du deep learning appliqué à l'imagerie médicale.
- Charger et prétraiter des images d'échographie.
- Construire un réseau de neurones convolutif (CNN) avec TensorFlow et Keras.
- Entraîner le modèle et évaluer ses performances.
- Visualiser les activations des couches du CNN pour comprendre son fonctionnement interne.
- Analyser les erreurs du modèle et identifier des pistes d'amélioration.
- Découvrir les bonnes pratiques pour le développement de modèles d'IA en santé.

## Prérequis

- Aucun ! Cet atelier est conçu pour les débutants.
- Un compte Google (pour utiliser Google Colab).

## Outils

- **Google Colab :** Un environnement de développement en ligne gratuit, qui ne nécessite aucune installation. Tout se passe dans votre navigateur !
- **TensorFlow et Keras :** Des librairies Python puissantes pour le deep learning.
- **Kaggle Hub :** Pour télécharger facilement le jeu de données.
- **OpenCV (cv2) :** Pour le prétraitement des images.
- **Matplotlib et NumPy :** Pour la visualisation et la manipulation des données.
- **Scikit-learn** Pour séparer le jeu de données.
- **Pandas** Pour manipuler les données.

## Jeu de Données (Dataset)

Nous utiliserons le jeu de données "Breast Ultrasound Images Dataset" disponible sur Kaggle : [https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

Ce dataset contient des images d'échographie de seins, classées en deux catégories :

- **Bénin :** Non cancéreux.
- **Malin :** Cancéreux.

## Structure du Notebook

1.  **Installation et Chargement des Données :**

    - Importation des librairies nécessaires.
    - Téléchargement du dataset depuis Kaggle.
    - Exploration de la structure du dataset.
    - Affichage d'exemples d'images.

2.  **Prétraitement et Nettoyage des Images :**

    - Lecture des images en niveaux de gris.
    - Redimensionnement à une taille standard (224x224).
    - Amélioration du contraste avec CLAHE (Contrast Limited Adaptive Histogram Equalization).
    - Normalisation des valeurs de pixels.
    - Ajout d'une dimension de canal.

3.  **Création du Modèle CNN :**

    - Définition de l'architecture du CNN (couches de convolution, pooling, couches denses, dropout).
    - Compilation du modèle (choix de l'optimiseur, de la fonction de perte et des métriques).
    - Explication des concepts clés (CNN, convolution, pooling, fonction d'activation, dropout, optimiseur, fonction de perte).

4.  **Entraînement du Modèle :**

    - Création du dataframe contenant le chemin des images, et leurs labels.
    - Séparation du jeu de données en données d'entrainement, et de validation.
    - Création du jeu de données Tensorflow.
    - Entraînement du modèle sur les données d'entraînement.
    - Suivi de la perte et de la précision pendant l'entraînement.
    - Explication du concept d'epoch.

5.  **Évaluation du Modèle :**

    - Calcul de la perte et de la précision sur les données de validation.

6.  **Exercices Pratiques (avec solutions) :**
    - **Exercice 1 : Augmentation de Données :**
      - Application de transformations aléatoires aux images (rotation, zoom, translation, luminosité, contraste).
      - Visualisation des images transformées.
      - Explication de l'intérêt de l'augmentation de données.
    - **Exercice 2 : Visualisation des Couches du Modèle :**
      - Création d'un modèle intermédiaire pour extraire les activations des couches.
      - Affichage des cartes d'activation des couches de convolution et de pooling.
      - Interprétation des activations.
    - **Exercice 3 : Visualisation des Prédictions et Confiance :**
      - Affichage d'images avec les prédictions du modèle, les vrais labels et les niveaux de confiance.
      - Modification du seuil de classification.
      - Identification des cas où le modèle est le moins confiant.
    - **Exercice 4 : Analyse des Erreurs en Fonction des Caractéristiques des Images :**
      - Calcul de métriques (flou, contraste, luminosité) sur les images.
      - Regroupement des images en fonction de ces métriques.
      - Calcul du taux d'erreur pour chaque groupe.
      - Visualisation des images de chaque catégorie.
    - **Exercice 5 : Analyse Qualitative des Erreurs :**
      - Identification manuelle des cas d'erreur.
      - Formulation d'hypothèses sur les causes des erreurs.
      - Discussion des pistes d'amélioration.

## Concepts Clés

- **Deep Learning :** Une branche de l'intelligence artificielle qui utilise des réseaux de neurones profonds pour apprendre à partir de données.
- **Réseau de Neurones Convolutif (CNN) :** Un type de réseau de neurones particulièrement adapté au traitement des images.
- **Convolution :** Une opération mathématique qui applique un filtre à une image pour en extraire des caractéristiques.
- **Pooling :** Une opération qui réduit la taille des données tout en conservant les informations les plus importantes.
- **Fonction d'Activation :** Une fonction non linéaire (ReLU, sigmoïde) qui introduit de la complexité dans le modèle.
- **Dropout :** Une technique de régularisation qui aide à prévenir le surapprentissage.
- **Optimiseur (Adam) :** Un algorithme qui ajuste les poids du réseau pendant l'entraînement.
- **Fonction de Perte (binary_crossentropy) :** Une fonction qui mesure l'erreur entre les prédictions du modèle et les vraies valeurs.
- **Epoch :** Un passage complet sur l'ensemble du dataset d'entraînement.
- **Augmentation de Données (Data Augmentation) :** Une technique qui consiste à appliquer des transformations aléatoires aux images pour augmenter artificiellement la taille du dataset.
- **Interprétabilité :** La capacité à comprendre _comment_ un modèle prend ses décisions.

## Pour Aller Plus Loin

- Explorer d'autres architectures de CNN (ResNet, Inception, EfficientNet).
- Utiliser des techniques d'interprétabilité plus avancées (Grad-CAM, SHAP).
- Collecter plus de données et améliorer la qualité des annotations.
- Collaborer avec des experts médicaux pour valider les résultats et identifier les biais potentiels.
- Déployer le modèle dans une application web ou mobile.
- Consulter la documentation des librairies.
