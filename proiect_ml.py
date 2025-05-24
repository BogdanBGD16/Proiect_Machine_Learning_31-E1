"""proiect_ml.py

Acest modul antrenează un model de clasificare și îi măsoară acuratețea

Proiect la disciplina Limbaje de Programare 3
UPT, ETcTI, Anul II, Seria B, Grupa 3, Semigrupa 3.1
Echipa E1 - Matica Bogdan Fabian și Maxim Marusia Diana

Surse: scikit-learn.com, kaggle.com, chatgpt.com

Mai 2025
"""

# Pasul 1. Descărcarea setului de date cu kagglehub (kaggle.com)
import kagglehub
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

# Pasul 2. Încărcarea fișierului CSV (chatgpt.com)
import pandas as pd
df = pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Pasul 3. Preprocesarea datelor (chatgpt.com)
df = df.dropna()
df = df.drop(['customerID'], axis=1)

# Pasul 4. Transformarea variabilelor categorice în valori numerice (chatgpt.com)
from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Pasul 5. Separarea variabilelor independente și țintă (chatgpt.com)
x = df.drop("Churn", axis=1)
y = df["Churn"]

# Pasul 6. Împărțirea în seturi de antrenare și test (chatgpt.com)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Pasul 7. Antrenarea modelului (scikit-learn.org)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model = model.fit(x_train, y_train)

# Pasul 8. Predicția rezulatelor pe setul de test (scikit-learn.org)
y_pred = model.predict(x_test)

# Pasul 9. Evaluarea predicției (scikit-learn.org)
from sklearn.metrics import balanced_accuracy_score
score = balanced_accuracy_score(y_test, y_pred)
score_percentage = score * 100

# Pasul 10. Afișarea rezultatului
print(f"Acuratețea modelului de clasificare este: {score:.4f}")
print(f"Acuratețea modelului de clasificare în procente este: {score_percentage:.2f}%")