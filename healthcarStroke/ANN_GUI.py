# Import libraries
import streamlit as st
import pandas as pd
import Lists as list
from joblib import load
import tensorflow as tf

# # import SVM Model and Scaler
# ann_model = load("ann.pkl")
scaler_model = load("scaler.pkl")


# Load the model from the file
ann_model = tf.keras.models.load_model('ann.h5')


gender_list = ['Male', 'Female', 'Other']
ever_married_list = ['Yes', 'No']
work_type_list = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
Residence_type_list = ['Urban', 'Rural']
smoking_status_list = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
#num list
hypertension_list=['Yes','No']
heart_disease_list=['Yes','No']
stroke_list=['NO','Yes']

# convert list to dictionary
# Industry_Sector_dict = {Industry_Sector_index: index for index, Industry_Sector_index in enumerate(list.Industry_Sector_list)}
gender_dict = {gender_index: index for index, gender_index in enumerate(gender_list)}
hypertension_dict = {hypertension_index: index for index, hypertension_index in enumerate(hypertension_list)}
heart_disease_dict = {heart_disease_index: index for index, heart_disease_index in enumerate(heart_disease_list)}
ever_married_dict = {ever_married_index: index for index, ever_married_index in enumerate(ever_married_list)}
work_type_dict = {work_type_index: index for index, work_type_index in enumerate(work_type_list)}
Residence_type_dict = {Residence_index: index for index, Residence_index in enumerate(Residence_type_list)}
smoking_status_dict = {smoking_status_index: index for index, smoking_status_index in enumerate(smoking_status_list)}

# gender-	age-	  hypertension-	 heart_disease-	 ever_married-	 work_type-	Residence_type	avg_glucose_level	bmi    	smoking_status
# Encoded Function
# def encode_fun(gender_class, age_class, hypertension_class, heart_disease_class, ever_married_class, work_type_class, Residence_type_class, avg_glucose_level_class, bmi_class, smoking_status_class):
#     gender_label = gender_dict[gender_class]
#     hypertension_label = hypertension_dict[hypertension_class]
#     heart_disease_label = heart_disease_dict[heart_disease_class]
#     ever_married_lable= ever_married_dict[ever_married_class]
#     work_type_label = work_type_dict[work_type_class]
#     Residence_type_label = Residence_type_dict[Residence_type_class]
#     smoking_status_label = smoking_status_dict[smoking_status_class]
#     return gender_label, age_class, hypertension_label, heart_disease_label, ever_married_lable, work_type_label, Residence_type_label, avg_glucose_level_class, bmi_class, smoking_status_label
  
# def decode_fen(stroke_state):
#     if(stroke_state==1.0):
#         return"He has Stroke"
#     else:
#         return"He has No Stroke"
    
# Scaler Function
def scale_fun(scaler, data):
    return scaler.transform(data) 

# GUI
st.title("Stroke State With ANN")

# Get user input for features
gender_class = st.selectbox("Gender",gender_list)
age_class = st.number_input("Age", min_value=0,max_value=82)
hypertension_class = st.selectbox("Hypertension ?",hypertension_list)
heart_disease_class = st.selectbox("Heart Disease ?",heart_disease_list)
ever_married_class = st.selectbox("Ever Married Level",ever_married_list)
work_type_class= st.selectbox("Work Type",work_type_list)
Residence_type_class = st.selectbox("Residence Type",Residence_type_list)
avg_glucose_level_class = st.number_input("Average Glucose Level", min_value=55,max_value=271)
bmi_class = st.number_input("BMI", min_value=10,max_value=97)
smoking_status_class = st.selectbox("Smoking Status",smoking_status_list)

# # Get user input for features
# gender_class = st.selectbox("Gender", list.gender_list)
# age_class = st.number_input("Age", min_value=0,max_value=82)
# hypertension_class = st.selectbox("Hypertension ?", list.hypertension_list)
# heart_disease_class = st.selectbox("Heart Disease ?", list.heart_disease_list)
# ever_married_class = st.selectbox("Ever Married Level", list.ever_married_list)
# work_type_class= st.selectbox("Work Type", list.work_type_list)
# Residence_type_class = st.selectbox("Residence Type", list.Residence_type_list)
# avg_glucose_level_class = st.number_input("Average Glucose Level", min_value=55,max_value=271)
# bmi_class = st.number_input("BMI", min_value=10,max_value=97)
# smoking_status_class = st.selectbox("Smoking Status", list.smoking_status_list)

# # Predict salary
# if st.button("Stroke State"):
#     # gender-	age-	  hypertension-	 heart_disease-	 ever_married-	 work_type-	Residence_type	avg_glucose_level	bmi	  smoking_status
#     # gender	age	      hypertension	heart_disease	ever_married	  work_type  Residence_type   avg_glucose_level   bmi	  smoking_status
#     encoded_data = encode_fun(gender_class, age_class, hypertension_class,  heart_disease_class, ever_married_class, work_type_class, Residence_type_class, avg_glucose_level_class, bmi_class, smoking_status_class)
#     data_to_classification = pd.DataFrame([encoded_data], columns=['gender','age','hypertension','heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
    
#     print(data_to_classification)

#     # Scale data
#     scaled_data = scale_fun(scaler_model, data_to_classification)
#     st.write(scaled_data)
#     print(scaled_data)
#     # stroke state
#     stroke_state = ann_model.predict(scaled_data)[0]
#     # st.write(stroke_state)
#     # stroke_state_decode=decode_fen(stroke_state)
#     if( stroke_state == 0 ):
#         st.success(f"Stroke State: He has No Stroke ")
#     else:
#         st.warning(f"Stroke State: He has Stroke ")
      
#     # st.success(f"Stroke State: {stroke_state}")
