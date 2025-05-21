import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load the dataset (Heart rate)
heart_data = pd.read_csv("heart.csv")

# Prepare features (X) and target (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Streamlit interface
st.title("Heart Disease Prediction System")

st.sidebar.title("Enter Input Values")
form = st.sidebar.form(key='my_form')

# Input fields for prediction
age = form.number_input(label="Age", min_value=0)
gender = form.radio("Gender", ["Male", "Female"])
cp = form.number_input(label="Chest Pain", min_value=0, max_value=3)
bp = form.number_input(label="Resting Blood Pressure", min_value=0)
chol = form.number_input(label="Serum Cholesterol", min_value=0)
fbs = form.number_input(label="Fasting Blood Sugar", min_value=0, max_value=1)
restecg = form.number_input(label="Resting Electrocardiographic Results", min_value=0, max_value=2)
thalch = form.number_input(label="Max Heart Rate", min_value=0)
exang = form.number_input(label="Exercise-induced Angina", min_value=0, max_value=1)
oldpeak = form.number_input(label="ST Depression", min_value=0.0)
slope = form.number_input(label="Slope", min_value=0, max_value=2)
ca = form.number_input(label="Number of Major Vessels", min_value=0, max_value=4)
thal = form.number_input(label="Thalassemia", min_value=0, max_value=3)

# Algorithm selection
select_algorithm = form.selectbox("Select Algorithm", ["KNN", "Decision Tree"])
submit_button = form.form_submit_button(label='Submit')

# Convert gender to numeric
gender = 1 if gender == "Male" else 0

# If the form is submitted
if submit_button:
    input_data = (int(age), int(gender), int(cp), int(bp), int(chol), int(fbs), int(restecg), int(thalch), int(exang), 
                  float(oldpeak), int(slope), int(ca), int(thal))
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Train the model based on selected algorithm
    if select_algorithm == "KNN":
        model = KNeighborsClassifier()
        model.fit(X_train, Y_train)
        prediction = model.predict(input_data_reshaped)

        # Calculate accuracy
        accuracy = accuracy_score(Y_test, model.predict(X_test))
        
        # Feature importance graph for KNN
        st.subheader("Input Data Graph for KNN")
        plt.figure(figsize=(10, 5))
        
        # Create a bar chart
        plt.bar(range(len(input_data)), input_data, tick_label=X.columns)
        plt.xticks(rotation=45)
        plt.ylabel("Input Value")
        plt.title("Input Data for KNN Prediction")
        
        # Display the graph in Streamlit
        st.pyplot(plt)

    elif select_algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        prediction = model.predict(input_data_reshaped)

        # Calculate accuracy
        accuracy = accuracy_score(Y_test, model.predict(X_test))

    # Display the prediction and accuracy
    if prediction[0] == 1:
        st.subheader("Result: Positive")
        st.write(f"Accuracy: {accuracy * 100:.2f}%")

        # Determine the most important feature using Random Forest for better insights
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, Y_train)
        feature_importances = rf_model.feature_importances_
        
        # Display the importance of features
        sorted_indices = np.argsort(feature_importances)[::-1]
        st.subheader("Feature Importance for Positive Prediction")
        importance_data = pd.DataFrame({
            'Feature': X.columns[sorted_indices],
            'Importance': feature_importances[sorted_indices]
        })
        
        # Show top important features
        st.table(importance_data.head(5))  # Display top 5 important features

    else:
        st.subheader("Result: Negative")
        st.write("No disease danger has been detected. You are healthy! 😊")
        st.write("**BE HAPPY, BE HEALTHY!**")
        st.write("**REGRADS: PAYAL, PRASTUTI, MAYURI, ABHIBHOO**")
