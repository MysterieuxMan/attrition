from ipywidgets import Dropdown, IntSlider, interact
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

from sklearn.base import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('employee_data.csv')

def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test,y_test_pred)
    testconfusion = metrics.confusion_matrix(y_test, y_test_pred)
    trainconfusion = metrics.confusion_matrix(y_train, y_train_pred)

    print("TRAINIG RESULTS: \n===============================")
    
    print(f"CONFUSION MATRIX:\n{trainconfusion}")
    print(f"ACCURACY SCORE:\n{acc_train:.4f}")
    print("precision score:", round(metrics.precision_score(y_train,y_train_pred),2))
    print("Recall Accuracy:", round(metrics.recall_score(y_train,y_train_pred),2))
    print("Area Under Curve AUC:", round(metrics.roc_auc_score(y_train,y_train_pred),2))
    
    print("\n\nTRAINIG RESULTS: \n===============================")
    
    print(f"CONFUSION MATRIX:\n{testconfusion}")
    print(f"ACCURACY SCORE:\n{acc_test:.4f}")
    print("precision score:", round(metrics.precision_score(y_test,y_test_pred),2))
    print("Recall Accuracy:", round(metrics.recall_score(y_test,y_test_pred),2))
    print("Area Under Curve AUC:", round(metrics.roc_auc_score(y_test,y_test_pred),2))
  
data = df.copy()
data.drop(columns="NumCompaniesWorkedGroup", inplace =True)
data["Attrition"]= data["Attrition"].replace({"Yes":1,
"No": 0})
X = data.drop(columns=['Attrition','HourlyRate','MonthlyRate',
                       'NumCompaniesWorked','PercentSalaryHike','YearsSinceLastPromotion',
                      'JobInvolvement','Education','Gender','YearsAtCompany','PerformanceRating','YearsWithCurrManager'], axis=1)
y = data.Attrition

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, 
                                                    stratify=y)# because the data is unbalanced
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train,y_train)

def make_prediction(age, businesstravel,dailyrate,department,distanceFromHome,
                    educationfield,environmentsatisfaction,
                    joblevel,jobrole,jobsatisfaction,maritalstatus,
                    monthlyincome,overtime,relationshipsatisfaction,
                    stockoptionlevel,totalworkingyears,trainingtimeslastyear,
                    worklifebalance,yearsincurrentrole
                   ):
    data={
        'Age':age,
        'BusinessTravel':businesstravel,
        'DailyRate':dailyrate,
         'Department':department,
        'DistanceFromHome' :distanceFromHome,
       'EducationField':educationfield,
        'EnvironmentSatisfaction':environmentsatisfaction,
        'JobLevel':joblevel,
        'JobRole':jobrole,
        'JobSatisfaction':jobsatisfaction,
        'MaritalStatus':maritalstatus,
       'MonthlyIncome':monthlyincome,
        'OverTime':overtime,
        'RelationshipSatisfaction':relationshipsatisfaction,
       'StockOptionLevel':stockoptionlevel,
        'TotalWorkingYears':totalworkingyears,
        'TrainingTimesLastYear':trainingtimeslastyear,
        'WorkLifeBalance':worklifebalance,
         'YearsInCurrentRole':yearsincurrentrole
    }

  
    df=pd.DataFrame(data,index=[0])
    prediction = model.predict_proba(df)[:, 1][0]
    if prediction > 0.4:
        Risk ="Strong"
    elif prediction > 0.3:
        Risk = "Medium"
    else:
        Risk ="Weak" 
    return f"This employee has a {Risk} Risk to quit with Probability = {round(prediction,5)}."



interact(
    make_prediction,
    age=IntSlider(min=X_train["Age"].min(), max=X_train["Age"].max(),value=X_train["Age"].mean()),
    businesstravel=Dropdown(options=sorted(X_train["BusinessTravel"].unique())),
    dailyrate=IntSlider(min=X_train["DailyRate"].min(), max=X_train["DailyRate"].max(),value=X_train["DailyRate"].mean()),    
    department=Dropdown(options=sorted(X_train["Department"].unique())),
    distanceFromHome=IntSlider(min=X_train["DistanceFromHome"].min(), max=X_train["DistanceFromHome"].max(),value=X_train["DistanceFromHome"].mean()),
    educationfield=Dropdown(options=sorted(X_train["EducationField"].unique())),
    environmentsatisfaction=Dropdown(options=sorted(X_train["EnvironmentSatisfaction"].unique())),
    joblevel=Dropdown(options=sorted(X_train["JobLevel"].unique())),
    jobrole=Dropdown(options=sorted(X_train["JobRole"].unique())),
    jobsatisfaction=Dropdown(options=sorted(X_train["JobSatisfaction"].unique())),
    maritalstatus=Dropdown(options=sorted(X_train["MaritalStatus"].unique())),
    monthlyincome=IntSlider(min=X_train["MonthlyIncome"].min(), max=X_train["MonthlyIncome"].max(),value=X_train["MonthlyIncome"].mean()),
    overtime=Dropdown(options=sorted(X_train["OverTime"].unique())),
    relationshipsatisfaction=Dropdown(options=sorted(X_train["RelationshipSatisfaction"].unique())),
    stockoptionlevel=Dropdown(options=sorted(X_train["StockOptionLevel"].unique())),
    totalworkingyears=IntSlider(min=X_train["TotalWorkingYears"].min(), max=X_train["TotalWorkingYears"].max(),value=X_train["TotalWorkingYears"].mean()),
    trainingtimeslastyear=IntSlider(min=X_train["TrainingTimesLastYear"].min(), max=X_train["TrainingTimesLastYear"].max(),value=X_train["TrainingTimesLastYear"].mean()),
    worklifebalance=Dropdown(options=sorted(X_train["WorkLifeBalance"].unique())),
    yearsincurrentrole=IntSlider(min=X_train["YearsInCurrentRole"].min(), max=X_train["YearsInCurrentRole"].max(),value=X_train["YearsInCurrentRole"].mean()),

);

Inputs = [
        gr.Slider(
           int(X_train["Age"].min()), int(X_train["Age"].max()),
           value=int(X_train["Age"].mean()),
            label="Age"
        ),
        gr.Dropdown(
            sorted(X_train["BusinessTravel"].unique()),
            value =X_train["BusinessTravel"].iloc[0],
            label="Business Travel"
        ),
        gr.Slider(
            int( X_train["DailyRate"].min() ), int(X_train["DailyRate"].max()),
            value=int(X_train["DailyRate"].mean()),
            label="Daily Rate"
        ),
        gr.Dropdown(
            sorted(X_train["Department"].unique()),
            value =X_train["Department"].iloc[0],
            label="Department"
        ),
        gr.Slider(
            int(X_train["DistanceFromHome"].min()),int( X_train["DistanceFromHome"].max()),
            value=int(X_train["DistanceFromHome"].mean()),
            label="Distance From Home"
        ),
        gr.Dropdown(
            sorted(X_train["EducationField"].unique()),
            value =X_train["EducationField"].iloc[0],
            label="Education Field"
        ),
        gr.Radio(
            sorted(X_train["EnvironmentSatisfaction"].unique()),
            value =X_train["EnvironmentSatisfaction"].iloc[0],
            label="Environment Satisfaction"
        ),
        gr.Dropdown(
            sorted(X_train["JobLevel"].unique()), 
            value =X_train["JobLevel"].iloc[0],
            label="Job Level"
        ),
        gr.Dropdown(
            sorted(X_train["JobRole"].unique()),
            value =X_train["JobRole"].iloc[0],
            label="Job Role"
        ),
        gr.Radio(
            sorted(X_train["JobSatisfaction"].unique()),
            value =X_train["JobSatisfaction"].iloc[0],
            label="Job Satisfaction"
        ),
        gr.Radio(
            sorted(X_train["MaritalStatus"].unique()),
            value =X_train["MaritalStatus"].iloc[0],
            label="Marital Status"
        ),
        gr.Slider(
            int(X_train["MonthlyIncome"].min()), int(X_train["MonthlyIncome"].max()),
            value=int(X_train["MonthlyIncome"].mean()),
            label="Monthly Income"
        ),
        gr.Radio(
            sorted(X_train["OverTime"].unique()),
            value =X_train["OverTime"].iloc[0],
            label="OverTime"
        ),
        gr.Radio(
            sorted(X_train["RelationshipSatisfaction"].unique()),
            value =X_train["RelationshipSatisfaction"].iloc[0],
            label="Relationship Satisfaction"
        ),
        gr.Radio(
            sorted(X_train["StockOptionLevel"].unique()),
            value =X_train["StockOptionLevel"].iloc[0],
            label="Stock Option Level"
        ),
        gr.Slider(
            int(X_train["TotalWorkingYears"].min()), int(X_train["TotalWorkingYears"].max()),
            value=int(X_train["TotalWorkingYears"].mean()), step =1,
            label="Total Working Years"
        ),
        gr.Slider(
            int(X_train["TrainingTimesLastYear"].min()),int( X_train["TrainingTimesLastYear"].max()),
            value=int(X_train["TrainingTimesLastYear"].mean()), step =1, 
            label="Training Times Last Year"
        ),
        gr.Radio(
            sorted(X_train["WorkLifeBalance"].unique()),
            value =X_train["WorkLifeBalance"].iloc[0],
            label="WorkLifeBalance"
        ),
        gr.Slider(
            int(X_train["YearsInCurrentRole"].min()), int(X_train["YearsInCurrentRole"].max()),
            value=int(X_train["YearsInCurrentRole"].mean()), step =1,
            label="Years in Current Role"
        ),

]

demo = gr.Interface( fn = make_prediction , inputs=Inputs, outputs="text", live = True, theme="soft")

demo.launch(share=True)