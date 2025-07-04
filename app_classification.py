import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title
st.title("Social Network Ad Prediction App")
st.write("This app predicts whether a user will buy a product based on Age and Estimated Salary.")

# Load Data
@st.cache_data
def load_data():
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y

X, y = load_data()

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Train Models
@st.cache_resource
def train_models():
    models = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf

    # SVM RBF
    svm_rbf = SVC(kernel='rbf', random_state=0)
    svm_rbf.fit(X_train_scaled, y_train)
    models['SVM (RBF)'] = svm_rbf

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train_scaled, y_train)
    models['KNN'] = knn

    # Logistic Regression
    lr = LogisticRegression(random_state=0)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr

    # SVM Linear
    svm_linear = SVC(kernel='linear', random_state=0)
    svm_linear.fit(X_train_scaled, y_train)
    models['SVM (Linear)'] = svm_linear

    return models

models = train_models()

# Sidebar inputs
st.sidebar.header("User Input Features")
age = st.sidebar.slider("Age", 18, 100, 30)
salary = st.sidebar.slider("Estimated Salary", 10000, 150000, 87000)
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))

# Predict button
if st.sidebar.button("Predict"):
    input_data = np.array([[age, salary]])
    input_scaled = sc.transform(input_data)

    selected_model = models[model_choice]
    prediction = selected_model.predict(input_scaled)[0]

    result = "Will Buy" if prediction == 1 else "Will NOT Buy"
    st.subheader(f"Prediction using {model_choice}:")
    st.success(result)

# Show model performance
if st.checkbox("Show Model Accuracy"):
    st.subheader("Model Accuracies on Test Set")
    perf_data = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        perf_data[name] = acc

    perf_df = pd.DataFrame(perf_data.items(), columns=['Model', 'Accuracy'])
    st.dataframe(perf_df.set_index('Model'))
