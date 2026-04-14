# 🩺 DiaPredict Pro AI
Application web de prédiction du risque de diabète, construite avec Streamlit et Scikit-learn, à partir du dataset Pima Indians Diabetes.

## 🏆 Contexte
Ce projet a été réalisé pour une compétition à l’UEMF par deux étudiants talentueux et bg:
- Esaie
- Charis

Nous avons travaillé sérieusement et en collaboration du début à la fin. Ce projet n’aurait pas été possible sans ce travail d’équipe rigoureux 🤝.

## 🎯 Objectif du projet
Créer une application interactive qui:
- estime la probabilité de risque de diabète 📈,
- affiche un diagnostic lisible (risque élevé ou faible) 🧠,
- explique les facteurs cliniques saisis avec des seuils de référence 🔍,
- propose des conseils concrets selon le profil utilisateur ✅.

## ✨ Fonctionnalités principales
- Prédiction du risque de diabète avec modèle ML.
- Interface multilingue (français, anglais, espagnol) 🌍.
- Seuil de décision configurable 🎚️.
- Explications détaillées des paramètres cliniques (glucose, tension, insuline, IMC, âge, etc.).
- QR code d’accès rapide à l’application 📲.
- Design moderne et expérience utilisateur fluide 🎨.

## 🖼️ Aperçu de l’application
![App main frame](https://github.com/Super-game17/DiaPredict-AI/blob/main/Streamlit-app-screenshot.png?raw=true)

## Fichiers clés du projet
- [app.py](app.py): application Streamlit principale.
- [requirements.txt](requirements.txt): dépendances Python.
- [Machine Learning project.ipynb](Machine%20Learning%20project.ipynb): entraînement, évaluation et export du modèle.
- [diabetes.csv](diabetes.csv): dataset principal utilisé pour l’apprentissage.

## 🚀 Lancer le projet en local
1. Cloner le repo
2. Créer et activer un environnement virtuel
3. Installer les dépendances:
   pip install -r requirements.txt
4. Lancer l’application:
   streamlit run app.py

## ☁️ Déploiement
L’application est déployable facilement sur Streamlit Community Cloud via GitHub.

## 👨‍💻 Auteurs
- Esaie
- Charis

## ⚠️ Remarque importante
Cette application est un outil d’aide à la décision et ne remplace pas un avis médical professionnel.
