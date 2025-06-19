import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle


conn = pymysql.connect(
    host='localhost',
    user='root',       
    password='Unimaginatively@099801',   
    database='loan_prediction'  
)

query = "SELECT * FROM loan_applications"
df = pd.read_sql(query, conn)
conn.close()


df.fillna(method='ffill', inplace=True)


label_cols = ['Gender', 'Married', 'Dependents', 'Education',
              'Self_Employed', 'Property_Area', 'Loan_Status']

for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])


X = df.drop(columns=['Loan_ID', 'Loan_Status']) 
y = df['Loan_Status']                           


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as loan_model.pkl")
