import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
train_df = pd.read_csv('Titanic_train.csv')
test_df = pd.read_csv('Titanic_test.csv')

st.title('Titanic Survival Prediction & Analysis')

# 1. Data Exploration
st.header('1. Data Exploration')
st.subheader('Training Data Sample')
st.write(train_df.head())

# 2. Data Preprocessing
st.header('2. Data Preprocessing')

# Fill missing Age values
# Fill missing Age values safely
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())


# Encode categorical variables
for df in [train_df, test_df]:
    df['Sex_male'] = (df['Sex'] == 'male').astype(int)
    df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
    df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)

features = ['Age', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = train_df[features]
y = train_df['Survived']

# 3. Model Building
st.header('3. Model Building')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
st.success('Model trained successfully.')

# 4. Model Evaluation
st.header('4. Model Evaluation')
y_pred = model.predict(X_val)
st.write('Accuracy:', accuracy_score(y_val, y_pred))
st.text('Classification Report:')
st.text(classification_report(y_val, y_pred))

# 5. Interpretation
st.header('5. Interpretation')
st.write('Model Coefficients:', dict(zip(features, model.coef_[0])))
st.write('Intercept:', model.intercept_[0])

# 6. Streamlit Deployment - Custom Input
st.header('6. Survival Prediction')
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sex = st.selectbox('Sex', ['male', 'female'])
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Create input dataframe
input_data = pd.DataFrame({
    'Age': [age],
    'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_C': [1 if embarked == 'C' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0]
})

if st.button('Predict'):
    prediction = model.predict(input_data)
    st.subheader('Survival Prediction:')
    st.success('Survived' if prediction[0] == 1 else 'Not Survived')