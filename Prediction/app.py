import pandas as pd
import numpy as np
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
with open(r'Prediction/Best_Random_Forest_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset for visualization
path = r"Data/synthetic_COPD_data.csv"
df = pd.read_csv(path)

# Streamlit App
def prediction_dashboard():
    st.title("COPD Prediction Dashboard")

    # User Input
    st.sidebar.header("User Input")
    
    age = st.sidebar.slider("Age", 30, 88, 50)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    bmi = st.sidebar.slider("BMI", 10, 40, 25)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["Current", "Former", "Never"])
    biomass_fuel_exposure = st.sidebar.selectbox("Biomass Fuel Exposure", ["Yes", "No"])
    occupational_exposure = st.sidebar.selectbox("Occupational Exposure", ["Yes", "No"])
    family_history = st.sidebar.selectbox("Family History", ["Yes", "No"])
    air_pollution_level = st.sidebar.slider("Air Pollution Level", 0, 300, 50)
    respiratory_infections = st.sidebar.selectbox("Respiratory Infections in Childhood", ["Yes", "No"])
    location = st.sidebar.selectbox("Location", ['Chitwan', 'Dharan', 'Hetauda', 'Kathmandu', 'Lalitpur', 'Nepalgunj', 'Pokhara'])

    # Process the input data
    input_data = {
        'Age': [age],
        'Gender': [gender],
        'Biomass_Fuel_Exposure': [biomass_fuel_exposure],
        'Occupational_Exposure': [occupational_exposure],
        'Family_History_COPD': [family_history],
        'BMI': [bmi],
        'Air_Pollution_Level': [air_pollution_level],
        'Respiratory_Infections_Childhood': [respiratory_infections],
        'Smoking_Status_encoded': [smoking_status],
        'Location': [location]
    }

    # Convert the data to a DataFrame
    input_df = pd.DataFrame(input_data)

    # Encode categorical features
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['Smoking_Status_encoded'] = input_df['Smoking_Status_encoded'].map({'Current': 1, 'Former': 0.5, 'Never': 0})
    input_df['Biomass_Fuel_Exposure'] = input_df['Biomass_Fuel_Exposure'].map({'Yes': 1, 'No': 0})
    input_df['Occupational_Exposure'] = input_df['Occupational_Exposure'].map({'Yes': 1, 'No': 0})
    input_df['Family_History_COPD'] = input_df['Family_History_COPD'].map({'Yes': 1, 'No': 0})
    input_df['Respiratory_Infections_Childhood'] = input_df['Respiratory_Infections_Childhood'].map({'Yes': 1, 'No': 0})

    # One-Hot Encoding for the Location variable
    location_one_hot = pd.get_dummies(input_df['Location'], drop_first=True)

    # Drop the original 'Location' column and append the one-hot encoded columns
    input_df = pd.concat([input_df.drop(columns=['Location']), location_one_hot], axis=1)

    # Ensure that all necessary columns are present, even if they are zero-filled
    location_columns = ['Location_Biratnagar', 'Location_Butwal', 'Location_Chitwan', 
                        'Location_Dharan', 'Location_Hetauda', 'Location_Kathmandu', 
                        'Location_Lalitpur', 'Location_Nepalgunj', 'Location_Pokhara']
    
    for col in location_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing location columns with 0

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.write("Prediction: COPD Detected")
        else:
            st.write("Prediction: No COPD Detected")

def visualization_dashboard():
    st.title("COPD Data Exploration")

    # Univariate Analysis
    plt.figure(figsize=(14, 8))
    
    # Age Distribution
    st.subheader("Age Distribution")
    sns.histplot(df['Age'], kde=True, color='blue')
    plt.title('Age Distribution')
    st.pyplot(plt)
    plt.clf()

    # BMI Distribution
    st.subheader("BMI Distribution")
    sns.histplot(df['BMI'], kde = True, color='green')
    plt.title('BMI Distribution')
    st.pyplot(plt)
    plt.clf()

    #Gender Distribution
    st.subheader("Gender Distribution")
    sns.countplot(x = 'Gender', data= df, palette='viridis')
    plt.title('Gender Distribution')
    st.pyplot(plt)
    plt.clf()

    # Smoking Status
    st.subheader("Smoking Status Distribution")
    sns.countplot(x = 'Smoking_Status', data = df, palette = 'Set1')
    plt.title('Smoking status Distribuition')
    st.pyplot(plt)
    plt.clf()

    # Bivariate Analysis
    plt.figure(figsize = (14, 10))

    # Age vs COPD Diagnosis
    st.subheader("Age vs COPD Diagnosis")
    sns.boxplot(x = 'COPD_Diagnosis', y = 'Age', data = df, palette = 'coolwarm')
    plt.title('Age vs COPD Diagnosis')
    st.pyplot(plt)
    plt.clf()

    # BMI vs COPD Diagnosis
    st.subheader("BMI vs COPD_Diagnosis")
    sns.boxplot(x = 'COPD_Diagnosis', y = 'BMI', data = df, palette = 'coolwarm')
    plt.title('BMI vs COPD_Diagnosis')
    st.pyplot(plt)
    plt.clf()

    # Smoking Status vs COPD Diagnosis
    st.subheader("Smoking Status vs COPD_Diagnosis")
    sns.boxplot(x= 'Smoking_Status', hue = 'COPD_Diagnosis', data= df, palette = 'Set2')
    plt.title('Smoking Status vs COPD_Diagnosis')
    st.pyplot(plt)
    plt.clf()

    # Count plot of COPD Diagnosis
    st.subheader("Count Plot of COPD Diagnosis")
    sns.countplot(x = 'COPD_Diagnosis', data = df, palette = 'viridis')
    plt.title('Count plot of COPD_Diagnosis')
    st.pyplot(plt)
    plt.clf()

    # Gender vs COPD Diagnosis
    st.subheader("Gender vs COPD Diagnosis Count")
    sns.countplot(x='Gender', hue='COPD_Diagnosis', data=df, palette='Set2')
    plt.title("Gender vs COPD Diagnosis Count")
    st.pyplot(plt)
    plt.clf()

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    data_corr = df[['Age', 'Biomass_Fuel_Exposure', 'Occupational_Exposure', 'Family_History_COPD', 'BMI', 'Air_Pollution_Level', 'Respiratory_Infections_Childhood', 'COPD_Diagnosis']]
    corr = data_corr.corr()
    sns.heatmap(corr, annot = True, cmap = 'coolwarm', fmt = '.2f')
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    plt.clf()

# Main app function
def main():
    st.sidebar.title("App Menu")
    app_mode = st.sidebar.selectbox("Choose the app mode",["Data Visualization","COPD Prediction"])

    if app_mode == "COPD Prediction":
        prediction_dashboard()
    elif app_mode == "Data Visualization":
        visualization_dashboard()
        
if __name__ == "__main__":
    main()