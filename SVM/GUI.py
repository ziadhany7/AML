# Import libraries
import streamlit as st
import pandas as pd
import Lists as list
from joblib import load

# import SVM Model and Scaler
svm_model = load("SVM.pkl")
scaler_model = load("SVM_Scale.pkl")

# convert list to dictionary
# Industry_Sector_dict = {Industry_Sector_index: index for index, Industry_Sector_index in enumerate(list.Industry_Sector_list)}
gender_dict = {gender_index: index for index, gender_index in enumerate(list.gender_list)}
jop_title_dict = {gender_index: index for index, gender_index in enumerate(list.jop_title_list)}
# race_dict = {race_index: index for index, race_index in enumerate(list.race_list)}
country_dict = {country_index: index for index, country_index in enumerate(list.country_list)}
senior_dict = {senior_index: index for index, senior_index in enumerate(list.senior_list)}

# Encoded Function
def encode_fun(Age_class,gender_class, education_level_class, jop_title_lable,years_of_exp_class, country_class, senior_class):
    gender_label = gender_dict[gender_class]
    jop_title_lable= jop_title_dict[jop_title_class]
    # race_label = race_dict[race_class]
    country_label = country_dict[country_class]
    senior_label = senior_dict[senior_class]
    return Age_class,gender_label, education_level_class, jop_title_lable, years_of_exp_class, country_label, senior_label

# Scaler Function
def scale_fun(scaler, data):
    return scaler.transform(data) 

# GUI
st.title("Salary Prediction")

# Get user input for features
Age_class = st.number_input("Age",min_value=22,max_value=62)
gender_class = st.selectbox("Gender", list.gender_list)
education_level_class = st.selectbox("Education level", list.education_level_list)
jop_title_class= st.selectbox("Job Title", list.jop_title_list)
years_of_exp_class = st.number_input("Years Of Experience",min_value=1,max_value=34)
country_class = st.selectbox("Country", list.country_list)
senior_class = st.selectbox("Senior", list.senior_list)

# Predict salary
if st.button("Predict Salary"):
    # incude
    encoded_data = encode_fun(Age_class, gender_class, education_level_class, jop_title_class, years_of_exp_class , country_class, senior_class)
    data_to_predict = pd.DataFrame([encoded_data], columns=['Age','Gender','Education Level','Job Title','Years of Experience', 'Country', 'Senior'])
    
    # Scale data
    scaled_data = scale_fun(scaler_model, data_to_predict)
    
    # Predict salary
    predicted_salary = svm_model.predict(scaled_data)[0]
    
    st.success(f"Predicted Salary: ${predicted_salary:.2f}")
