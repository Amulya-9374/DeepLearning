# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack

# Load pre-trained models and transformers
rf_model = joblib.load('churnrf_model.pkl')
dnn_model = tf.keras.models.load_model('churndnn_model.keras')
imputer_numerical = joblib.load('churnimputer_numerical.pkl')
imputer_categorical = joblib.load('churnimputer_categorical.pkl')
encoder = joblib.load('churnencoder.pkl')
scaler = joblib.load('churnscaler.pkl')
svd = joblib.load('churnsvd.pkl')

# List of numerical and categorical columns
numerical_columns = ['CR_PROD_CNT_IL', 'AMOUNT_RUB_CLO_PRC', 'PRC_ACCEPTS_A_EMAIL_LINK', 
                     'APP_REGISTR_RGN_CODE', 'PRC_ACCEPTS_A_POS', 'CLNT_SALARY_VALUE',
                     'TURNOVER_DYNAMIC_IL_1M', 'AMOUNT_RUB_SUP_PRC', 'REST_DYNAMIC_FDEP_1M', 
                     'REST_DYNAMIC_SAVE_3M', 'CR_PROD_CNT_VCU', 'REST_AVG_CUR', 'AMOUNT_RUB_NAS_PRC',
                     'TRANS_COUNT_SUP_PRC', 'TRANS_COUNT_NAS_PRC', 'CR_PROD_CNT_TOVR', 
                     'CR_PROD_CNT_PIL', 'TURNOVER_CC', 'TRANS_COUNT_ATM_PRC', 'AMOUNT_RUB_ATM_PRC', 
                     'TURNOVER_PAYM', 'AGE', 'CR_PROD_CNT_CC', 'REST_DYNAMIC_FDEP_3M', 'REST_DYNAMIC_IL_1M', 
                     'CR_PROD_CNT_CCFP', 'REST_DYNAMIC_CUR_1M', 'REST_AVG_PAYM', 'LDEAL_GRACE_DAYS_PCT_MED', 
                     'REST_DYNAMIC_CUR_3M', 'CNT_TRAN_SUP_TENDENCY3M', 'TURNOVER_DYNAMIC_CUR_1M', 
                     'REST_DYNAMIC_PAYM_3M', 'SUM_TRAN_SUP_TENDENCY3M', 'REST_DYNAMIC_IL_3M', 
                     'CNT_TRAN_ATM_TENDENCY3M', 'CNT_TRAN_ATM_TENDENCY1M', 'TURNOVER_DYNAMIC_IL_3M', 
                     'SUM_TRAN_ATM_TENDENCY3M', 'SUM_TRAN_ATM_TENDENCY1M', 'REST_DYNAMIC_PAYM_1M', 
                     'TURNOVER_DYNAMIC_CUR_3M', 'CLNT_SETUP_TENOR', 'TURNOVER_DYNAMIC_PAYM_3M', 
                     'TURNOVER_DYNAMIC_PAYM_1M', 'TRANS_AMOUNT_TENDENCY3M', 'TRANS_CNT_TENDENCY3M', 
                     'REST_DYNAMIC_CC_1M', 'TURNOVER_DYNAMIC_CC_1M', 'REST_DYNAMIC_CC_3M', 
                     'TURNOVER_DYNAMIC_CC_3M', 'CNT_TRAN_AUT_TENDENCY1M']


categorical_columns = ['APP_MARITAL_STATUS', 'APP_KIND_OF_PROP_HABITATION', 'CLNT_JOB_POSITION', 
                       'CLNT_TRUST_RELATION', 'APP_DRIVING_LICENSE', 'APP_EDUCATION', 
                       'APP_POSITION_TYPE', 'APP_EMP_TYPE', 'APP_COMP_TYPE', 'PACK']

# Load training data to get unique values for categorical features
data = pd.read_csv("bank_data_train.csv")

# Ensure only existing columns are used
existing_numerical_columns = [col for col in numerical_columns if col in data.columns]
existing_categorical_columns = [col for col in categorical_columns if col in data.columns]

# Define a function to preprocess the input features
def preprocess_features(features):
    features_df = pd.DataFrame([features], columns=existing_numerical_columns + existing_categorical_columns)
    
    # Check for missing columns
    missing_cols = set(existing_numerical_columns + existing_categorical_columns) - set(features_df.columns)
    for col in missing_cols:
        features_df[col] = 0
    
    # Apply the same preprocessing steps as during training
    features_df[existing_numerical_columns] = imputer_numerical.transform(features_df[existing_numerical_columns])
    features_df[existing_categorical_columns] = imputer_categorical.transform(features_df[existing_categorical_columns])
    
    encoded_categorical = encoder.transform(features_df[existing_categorical_columns])
    features_numerical = pd.DataFrame(features_df[existing_numerical_columns], columns=existing_numerical_columns)
    features_combined = hstack([features_numerical, encoded_categorical])
    features_combined = scaler.transform(features_combined)
    features_combined = svd.transform(features_combined)
    
    return features_combined

# Streamlit app
st.title("Model Prediction App")
st.write("This app uses a Random Forest and a DNN model to make predictions.")

# Input fields for numerical features
numerical_inputs = {}
for i, col in enumerate(existing_numerical_columns):
    numerical_inputs[col] = st.number_input(f"Enter value for {col}", key=f"num_input_{i}")

# Input fields for categorical features
categorical_inputs = {}
for i, col in enumerate(existing_categorical_columns):
    categorical_inputs[col] = st.selectbox(f"Select {col}", options=data[col].unique(), key=f"cat_input_{i}")

# Combine all inputs into a single dictionary
all_inputs = {**numerical_inputs, **categorical_inputs}

if st.button("Predict"):
    # Preprocess the input features
    features = preprocess_features(all_inputs)
    
    # Make predictions using the RandomForest model
    rf_prediction = rf_model.predict(features)
    st.write(f"Random Forest Prediction: {rf_prediction[0]}")
    
    # Make predictions using the DNN model
    dnn_prediction = (dnn_model.predict(features) > 0.5).astype("int32")
    st.write(f"DNN Prediction: {dnn_prediction[0][0]}")

