import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
# Crearea unui settttt deeeee date fictiv pe baza structurii furnizatessssssssssssss
np.random.seed(42)  # pentru reproductibilitatedd
data = {
   'REPORTING_DATE': pd.date_range(start='2023-01-01', periods=1000, freq='D'),
   'CUST_ID': np.arange(1000),
   'AGE': np.random.randint(18, 70, size=1000),
   'GENDER': np.random.choice(['M', 'F'], size=1000),
   'NO_DEBIT_CARDS': np.random.randint(0, 5, size=1000),
   'NO_CREDIT_CARDS': np.random.randint(0, 5, size=1000),
   'BRANCH_CODE': np.random.randint(1, 100, size=1000),
   'NO_OF_TRX_12_MONTH': np.random.randint(0, 100, size=1000),
   'START_RELATIONSHIP': pd.date_range(start='2000-01-01', periods=1000, freq='D'),
   'SALARY': np.random.randint(20000, 150000, size=1000),
   'NO_PRODUCTS': np.random.randint(1, 10, size=1000),
   'CUSTOMER_RELATIONSHIP_STATUS': np.random.choice(['Active', 'Inactive'], size=1000),
   'NO_TRX_POS': np.random.randint(0, 50, size=1000),
   'NO_TRX_ONLINE': np.random.randint(0, 50, size=1000),
   'CHURN': np.random.choice([0, 1], size=1000)
}
df = pd.DataFrame(data)
# Afișează primele rânduri pentru a verifica datele
print(df.head())
# Distribuția valorilor în coloana CHURN
sns.countplot(data=df, x='CHURN')
plt.show()
# Statistici descriptive pentru date
print(df.describe())
# Distribuția clienților în funcție de vârstă și churn
sns.histplot(data=df, x='AGE', hue='CHURN', multiple='stack')
plt.show()
# Eliminați coloanele neutilizabile (dacă este necesar)
df = df.drop(columns=['REPORTING_DATE', 'CUST_ID', 'START_RELATIONSHIP', 'BRANCH_CODE'])
# Tratarea valorilor lipsă
df.fillna(0, inplace=True)
# Convertiți coloanele categorice în numerice
label_encoder = LabelEncoder()
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])
df['CUSTOMER_RELATIONSHIP_STATUS'] = label_encoder.fit_transform(df['CUSTOMER_RELATIONSHIP_STATUS'])
# Caracteristici și etichete
X = df.drop(columns=['CHURN'])
y = df['CHURN']
# Împărțirea datelor în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardizarea caracteristicilor
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Construirea modelului RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predicția pe setul de testare
y_pred = model.predict(X_test)
# Evaluarea modelului
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
# Funcție pentru a face predicții pentru clienți noi
def predict_churn(new_data):
   # Preprocesarea noilor date
   new_data = new_data.drop(columns=['REPORTING_DATE', 'CUST_ID', 'START_RELATIONSHIP', 'BRANCH_CODE'])
   new_data.fillna(0, inplace=True)
   new_data['GENDER'] = label_encoder.transform(new_data['GENDER'])
   new_data['CUSTOMER_RELATIONSHIP_STATUS'] = label_encoder.transform(new_data['CUSTOMER_RELATIONSHIP_STATUS'])
   new_data = scaler.transform(new_data)
   # Predicția churn-ului
   predictions = model.predict(new_data)
   return predictions
# Exemplu de utilizare a funcției de predicție
new_client_data = pd.DataFrame({
   'REPORTING_DATE': ['2023-07-01'],
   'CUST_ID': [1001],
   'AGE': [35],
   'GENDER': ['M'],
   'NO_DEBIT_CARDS': [2],
   'NO_CREDIT_CARDS': [1],
   'BRANCH_CODE': [10],
   'NO_OF_TRX_12_MONTH': [45],
   'START_RELATIONSHIP': ['2010-05-15'],
   'SALARY': [75000],
   'NO_PRODUCTS': [3],
   'CUSTOMER_RELATIONSHIP_STATUS': ['Active'],
   'NO_TRX_POS': [20],
   'NO_TRX_ONLINE': [15],
   'CHURN': [0]  # Nu va fi folosită pentru predicție, doar pentru consistența structurii
})
# Prevede churn-ul pentru noii clienți
churn_predictions = predict_churn(new_client_data)
print(f"Churn prediction for the new client: {churn_predictions}")