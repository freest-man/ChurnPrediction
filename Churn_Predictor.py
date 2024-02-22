import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import streamlit as st
from PIL import Image
import time


logo = "banking.png"

st.set_page_config(
    page_title="ABC Bank",
    page_icon= logo,
    layout="centered"
)

col1, col2 = st.columns([1,10])  # Adjust the column ratios as needed


with col1:
    st.image(logo, width=50)

with col2:
    st.header('ABC Bank')



st.title('Welcome to CCP Tool') 
    


st.markdown("""
* The CCP tools helps in predicting how likely is an existing customer is to churn. 
* The tools uses the bank’s database of customers to train models which help in predicting.  
* You can check for the probability of exit                         
""")
st.title('Upload the csv') 

url = "https://raw.githubusercontent.com/freest-man/ChurnPrediction/main/train.csv"

st.markdown(""" *[Testing Data](https://raw.githubusercontent.com/freest-man/ChurnPrediction/main/train.csv) 
                *[Training Data](https://raw.githubusercontent.com/freest-man/ChurnPrediction/main/test.csv)
""")


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    

    if st.button("Run Churn Prediction"):
        train_data = pd.read_csv(url)
        test_data = dataframe

        with st.spinner('Please Wait...'):
            time.sleep(5)
        

        # Separate features and target variable
        X = train_data.drop(['id', 'Exited'], axis=1)
        y = train_data['Exited']

        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # Create transformers for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create the full pipeline with the logistic regression model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])

        # Train the model
        model.fit(X_train, y_train)

        # Predict probabilities on the validation set
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Evaluate the model using ROC AUC
        roc_auc = roc_auc_score(y_val, y_val_pred_proba)
        st.write(f'ROC AUC of the model: {roc_auc}')

        # Now, make predictions on the test set
        test_predictions_proba = model.predict_proba(test_data)[:, 1]

        Output = pd.DataFrame({'id': test_data['id'],'Surname':test_data['Surname']   ,'Exited': test_predictions_proba})
        
        Output = Output.sort_values(by='Exited', ascending=False)

        st.write(Output)

