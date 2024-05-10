# Import libraries
import streamlit as st
import pandas as pd
import Lists as list
from joblib import load
from tensorflow.keras.models import load_model

# import SVM Model and Scaler
# ann = load("ann.pkl")
# ann = load_model("ann.h5")
scaler_model = load("scaler.pkl")

#   Data in Lists
gender_list = ['Male', 'Female', 'Other']
ever_married_list = ['Yes', 'No']
work_type_list = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
Residence_type_list = ['Urban', 'Rural']
smoking_status_list = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
#num list
hypertension_list=['Yes','No']
heart_disease_list=['Yes','No']
stroke_list=['NO','Yes']

#Data in Dictionary
gender_dict = {'Male':0, 'Female':1, 'Other':2}
ever_married_dict = {'Yes':0, 'No':1}
work_type_dict = {"Private":0, "Self-employed":1, "Govt_job":2, "children":3, "Never_worked":4}
Residence_type_dict = {"Urban":0, "Rural":1}
smoking_status_dict = {"formerly smoked":0, "never smoked":1, "smokes":2, "Unknown":3}
hypertension_dict={"Yes":0,"No":1}
heart_disease_dict={"Yes":0,"No":1}

def encode_fun(gender_class, age_class, hypertension_class, heart_disease_class, ever_married_class, work_type_class, Residence_type_class, avg_glucose_level, bmi_class, smoking_status_class):
    gender_label = gender_dict[gender_class]
    hypertension_label = hypertension_dict[hypertension_class]
    heart_disease_label = heart_disease_dict[heart_disease_class]
    ever_married_lable= ever_married_dict[ever_married_class]
    work_type_label = work_type_dict[work_type_class]
    Residence_type_label = Residence_type_dict[Residence_type_class]
    smoking_status_label = smoking_status_dict[smoking_status_class]
    return gender_label, age_class, hypertension_label, heart_disease_label, ever_married_lable, work_type_label, Residence_type_label, avg_glucose_level, bmi_class, smoking_status_label

def scale_fun(scaler_, data):
    return scaler_.transform(data) 

# GUI
st.title("Stroke State With ANN")


# Get user input for features
gender_class = st.selectbox("Gender",gender_list)
age_class = st.number_input("Age", min_value=0,max_value=82)
hypertension_class = st.selectbox("Hypertension ?",hypertension_list)
heart_disease_class = st.selectbox("Heart Disease ?", list.heart_disease_list)
ever_married_class = st.selectbox("Ever Married Level", list.ever_married_list)
work_type_class= st.selectbox("Work Type", list.work_type_list)
Residence_type_class = st.selectbox("Residence Type", list.Residence_type_list)
avg_glucose_level_class = st.number_input("Average Glucose Level", min_value=55,max_value=271)
bmi_class = st.number_input("BMI", min_value=10,max_value=97)
smoking_status_class = st.selectbox("Smoking Status", list.smoking_status_list)

# Predict salary
if st.button("Stroke State With ANN"):
    # gender-	age-	  hypertension-	 heart_disease-	 ever_married-	 work_type-	Residence_type	avg_glucose_level	bmi	  smoking_status
    encoded_data = encode_fun(gender_class, age_class, hypertension_class,  heart_disease_class, ever_married_class, work_type_class, Residence_type_class, avg_glucose_level_class, bmi_class, smoking_status_class)
    data_to_classification= pd.DataFrame([encoded_data], columns=['gender','age','hypertension','heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
    # print(data_to_classification)
    # Scale data
    scaled_data = scale_fun(scaler_model, data_to_classification)
    # stroke state
    stroke_state = ann.predict(scaled_data)[0][0]
    # st.write(stroke_state)
    # stroke_state_decode=decode_fen(stroke_state)
    if( stroke_state == 0 ):
        st.success(f"Stroke State: He has No Stroke ")
    else:
        st.warning(f"Stroke State: He has Stroke ")
        
    # st.success(f"Stroke State: {stroke_state}")

