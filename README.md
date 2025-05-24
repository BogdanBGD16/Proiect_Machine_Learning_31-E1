# 📊 Predicția clienților care renunță la contract (Customer Churn Prediction)

Acest proiect folosește un algoritm de Machine Learning pentru a prezice dacă un client va renunța sau nu la un contract cu un operator de comunicații.

Scopul proiectului este măsurarea acurateței modelului antrenat cu algoritmul **Decision Tree** din biblioteca scikit-learn.

Rulează programul: `python proiect_ml.py`

---

## 🗂️ Setul de Date

Datele provin de pe kaggle.com:
[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Setul conține informații despre clienți pe coloane: vârstă, durata contractului, metoda de plată, costuri lunare, etc. Coloana țintă este Churn, care indică dacă un client a renunțat (Yes) sau nu (No).

---

## 🧮 Algoritm utilizat

-  Decision Tree Classifier (sklearn.tree.DecisionTreeClassifier)
-  Balanced Accuracy Score (sklearn.metrics.balanced_accuracy_score)

---

## 📝️ Pașii realizați în proiect

1. Descărcarea datasetului de pe Kaggle folosind *kagglehub*
2. Încărcarea datelor într-un DataFrame *Pandas*
3. Preprocesarea datelor:
   - Eliminarea valorilor lipsă (*dropna*)
   - Eliminarea coloanei *customerID* (neinformativă)
4. Transformarea coloanelor de tip text în valori numerice cu  ajutorul *LabelEncoder*
5. Separarea variabilelor *țintă*
6. Împărțirea în seturi de antrenare și test cu *train_test_split*
7. Antrenarea modelului de tip *arbore de decizie*
8. Predicția rezultatelor pe setul de test
9. Evaluarea performanței modelului cu *balanced_accuracy _score*
10. Afișarea rezultatelor

---

## 🧪 Rezultate

Modelul de clasificare a obținut o acuratețe de aproximativ: **0.6405 (64.05%)**

---

## 👥 Autori

**Echipa:** 31-E1 _(Tema D7-T1)_
- Matica Bogdan-Fabian, _ETcTI / IIB / 3.1_
- Maxim Marusia-Diana, _ETcTI / IIB / 3.1_

---

## 🔗 Surse și referințe

- Cod inspirat din documentațiile oficiale Scikit-learn și Kaggle: 
   - https://scikit-learn.org/stable/modules/tree.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
   - https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data
- Asistență tehnică, exemple de cod și explicații oferite de ChatGPT (OpenAI), mai 2025
