# import pandas as pd
# import numpy as np
# import streamlit as st
# import joblib
# import shap 
# import lime 

# from sklearn.compose import ColumnTransformer,make_column_selector
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import Pipeline
# from lime.lime_tabular import LimeTabularExplainer

# import os
# import config 


# # Create preprocessing pipeline
# def create_preprocessing_pipeline():
    
#     # Select numeric and categorical columns
#     num_cols = make_column_selector(dtype_include = 'number')
#     cat_cols = make_column_selector(dtype_include = 'object')
    
#     # Instantiate the transformers
#     scaler = StandardScaler()
#     ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     knn_imputer = KNNImputer(n_neighbors=2, weights='uniform')
    
#     # Create pipeline
#     num_pipe = Pipeline([
#         ('imputer', knn_imputer),
#         ('scaler', scaler),
#     ])
    
#     cat_pipe = Pipeline([
#         ('encoder', ohe)
#     ])
    
#     preprocessor = ColumnTransformer([
#         ('numeric', num_pipe, num_cols),
#         ('categorical', cat_pipe, cat_cols),
#     ], remainder='drop')
    
#     return preprocessor

# # Categorical encoder
# def categorical_encoder(df):
#     for column in df.select_dtypes(include=['object']).columns:
#         df[column] = pd.Categorical(df[column]).codes

#     return df 

# # Define st_shap to embed SHAP plots in streamlit
# def st_shap(plot):
#     shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#     st.components.v1.html(shap_html)


# @st.cache_resource
# # Load models
# def load_models():
#     lgbm_model = joblib.load(config.MODEL_LGBM_PATH)
#     DEC_model = joblib.load(config.MODEL_DEC_TREE_PATH)

#     return lgbm_model,DEC_model

# @st.cache_data

# # Load and fit preprocessor
# def load_and_fit_preprocessor(file_path):
#     flight_df = pd.read_csv(file_path)
#     preprocessor = create_preprocessing_pipeline()
#     preprocessor.fit(flight_df)
    
#     return preprocessor, flight_df

# # Collect user inputs
# def user_inpute_features():
#     inputs = {'Online boarding': st.sidebar.slider('Online boarding',1,5,3),
#         'Inflight wifi service':st.sidebar.slider('Inflight WIFI service',1,5,3),
#         'Inflight entertainment':st.sidebar.slider('Inflight entertainment',1,5,3),
#             'Checkin service':st.sidebar.slider('Checkin service',1,5,3),
            
#             'Seat comfort':st.sidebar.slider('Seat comfort',1,5,3),
#             'Age':st.sidebar.slider.number_input('Age',0,100,18),
#             'Flight Distance':st.sidebar.number_input('Flight Distance',0,10000,100),
#             'Business Travel':st.sidebar.selectbox('Business Travel',['Yes','No']),
#             'Loyal Customer':st.sidebar.selectbox('Loyal Customer',['Yes','No']),
#             'Class':st.sidebar.selectbox('Class',['Eco,''Eco Plus','Business'])}

#     return pd.DataFrame(inputs, index=[0])

# # Make predictions
# def make_predictions(model, preprocessor, input_df,explainer,lime_explainer):
#     try:
#         input_df_transformed = preprocessor.transform(categorical_encoder(input_df))
#         prediction =model.predict(input_df_transformed)

#         # Get SHAP values
#         shap_values = explainer.shap_values(input_df_transformed)

#         # If shap_values is a list, use appropiate class
#         if isinstance(shap_values, list):
#             shap_values = shap_values[1] # for binary classification, use second class


#         # Lime explanation
#         lime_explanation = lime_explainer.explain_instance(
#             data_row = input_df_transformed[0],
#             predict_fn = model.predict_proba,
#             num_features = config.LIME_NUM_FEATURES

#         )

#         return 'Satisfied' if prediction[0] else 'Not Satisfied', shap_values,explainer.expected_values, input_df_transformed, lime_explanation
    
#     except Exception as error:
#         st.error(f'Error occured during prediction: {str(error)}')

#         return None, None,None,None, None
    
# # Display logo and hero logo
# st.sidebar.image('images/logo.jpg', width=100)
# st.image('images/people.jpg', use_column_width=True)

# st.title('Cebu Pacific Customer Flight Satisfaction')
# st.header('Please tell us about your experience')


# # Load models and preprocessor

# lgbm_model , dec_tree =load_models()
# preprocessor , flight_df = load_and_fit_preprocessor(config.DATA_PATH)


# # Model Selection
# model_choice = st.sidebar.selectbox('Choose Model',['LightGBM', 'Decision Tree'])   
# selected_model = lgbm_model if model_choice == 'LightGBM' else dec_tree

# # Initialize SHAP and LIME
# explainer = shap.TreeExplainer(selected_model)
# lime_explainer = LimeTabularExplainer(
#     training_data= preprocessor.transform(categorical_encoder(flight_df))
#     feature_names= flight_df.columns,
#     class_names=['Not Satisfied','Satisfied'],
#     mode= 'classification'
# )

# # Collect user inputs
# st.sidebar.header('User Input')
# input_df = user_inpute_features()

# # Display user inputs
# st.write('### User Inputs')
# for key, value in input_df.iloc[0].items():
#     st.write(f'**{key}**:{value}')   

# # predict button
# if st.button('Make Prediction'):
#     result, shap_values,expected_value, input_df_transformed, lime_explanation=  make_predictions(selected_model,preprocessor,input_df, explainer, lime_explainer)  
    
#     if result:
#         st.write(f'### Prediction : {result}')

#         # Display SHAP values
#         st.write(f'SHAP force plot')
#         st_shap(shap.force_plot(explainer.expected_value[0],shap_values[0], input_df_transformed[0]))

#         # Display LIME explanation
#         st.write(f'###Lime explanation :')
#         lime_html = lime_explanation.as_html()
#         st.components.v1.html(lime_html)


import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import lime

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer

import os
import config

# Create preprocessing pipeline
def create_preprocessing_pipeline():
    
    # Select numeric and categorical columns
    num_cols = make_column_selector(dtype_include = 'number')
    cat_cols = make_column_selector(dtype_include = 'object')
    
    # Instantiate the transformers
    scaler = StandardScaler()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    knn_imputer = KNNImputer(n_neighbors=2, weights='uniform')
    
    # Create pipeline
    num_pipe = Pipeline([
        ('imputer', knn_imputer),
        ('scaler', scaler),
    ])
    
    cat_pipe = Pipeline([
        ('encoder', ohe)
    ])
    
    preprocessor = ColumnTransformer([
        ('numeric', num_pipe, num_cols),
        ('categorical', cat_pipe, cat_cols),
    ], remainder='drop')
    
    return preprocessor

# Catergorical encoder
def categorical_encoder(df):
    for column in df.select_dtypes(include=['object']).columns:

        df[column] = pd.Categorical(df[column]).codes

    return df

# Define st_shap to embed SHAP plots in streamlit
def st_shap(plot):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html)

@st.cache_resource
# Load models
def load_models():
    lgbm_model = joblib.load(config.MODEL_LGBM_PATH)
    dec_tree = joblib.load(config.MODEL_DEC_TREE_PATH)
    
    return lgbm_model, dec_tree

@st.cache_data
# Load and fit the preprocessor
def load_and_fit_preprocessor(file_path):
    flight_df = pd.read_csv(file_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(flight_df)
    
    return preprocessor, flight_df

# Collect user inputs
def user_input_features():
    inputs = {
        'Online boarding': st.sidebar.slider('Online boarding', 1, 5, 3),
        'Inflight wifi service': st.sidebar.slider('Inflight WIFI service', 1, 5, 3),
        'Inflight entertainment': st.sidebar.slider('Inflight Entertainment', 1, 5, 3),
        'Checkin service': st.sidebar.slider('Check-in Service', 1, 5, 3),
        'Seat comfort': st.sidebar.slider('Seat Comfort', 1, 5, 3),
        'Age': st.sidebar.number_input('Age', 0, 100, 18),
        'Flight Distance': st.sidebar.number_input('Flight Distance', 0, 10000, 100),
        'Business Travel': st.sidebar.selectbox('Business Travel', ['Yes', 'No']),
        'Loyal Customer': st.sidebar.selectbox('Loyal Customer', ['Yes', 'No']),
        'Class': st.sidebar.selectbox('Class', ['Eco', 'Eco Plus', 'Business'])
    }
    
    return pd.DataFrame(inputs, index=[0])


# Make predictions
def make_predictions(model, preprocessor, input_df, explainer, lime_explainer):
    try:
        input_df_transformed = preprocessor.transform(categorical_encoder(input_df))
        prediction = model.predict(input_df_transformed)
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_df_transformed)
        
        # If shap_values is a list, use appropriate class
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # for binary classification, use second class
        
        # LIME explanation
        lime_explanation = lime_explainer.explain_instance(
            data_row = input_df_transformed[0],
            predict_fn = model.predict_proba,
            num_features = config.LIME_NUM_FEATURES
        )
        
        return 'Satisfied' if prediction[0] else 'Not Satisfied', shap_values, explainer.expected_value, input_df_transformed, lime_explanation
        
    except Exception as error:
        st.error(f'Error occurred during prediction: {str(error)}')
        
        return None, None, None, None, None
    
# Display logo and hero image
st.sidebar.image('images/logo.jpg', width=100)
st.image('images/people.jpg', use_column_width=True)

st.title('Cebu Pacific Customer Flight Satisfaction')
st.header('Please Tell Us About Your Experience')

# Load the models and preprocessor
lgbm_model, dec_tree = load_models()
preprocessor, flight_df = load_and_fit_preprocessor(config.DATA_PATH)

# Model selection
model_choice = st.sidebar.selectbox('Choose Model', ['LightGBM', 'Decision Tree'])
selected_model = lgbm_model if model_choice == 'LightGBM' else dec_tree

# Initialize SHAP and LIME
explainer = shap.TreeExplainer(selected_model)
lime_explainer = LimeTabularExplainer(
    training_data = preprocessor.transform(categorical_encoder(flight_df)),
    feature_names = flight_df.columns,
    class_names = ['Not Satisfied', 'Satisfied'],
    mode = 'classification'
)

# Collect user inputs
st.sidebar.header('User Input')
input_df = user_input_features()

# Display user inputs
st.write('### User Inputs')
for key, value in input_df.iloc[0].items():
    st.write(f'{key}: {value}')
    
# Predict button
if st.button('Make Prediction'):
    result, shap_values, expected_value, input_df_transformed, lime_explanation = make_predictions(
        selected_model, preprocessor, input_df, explainer, lime_explainer
    )
    
    if result:
        st.write(f'### Prediction: {result}')
        
       
        
        # Display LIME explainer
        st.write('### LIME Explanation:')
        lime_html = lime_explanation.as_html()
        st.components.v1.html(lime_html)
        
