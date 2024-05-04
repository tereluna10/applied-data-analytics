"""
Maria Teresa Luna Plascencia
Class: CS 677
Date: 3/19/2024
Final Project: Loan Approval Predictor - Evaluation of various ML models
Helper library that contains a set of functions that are used across the final
project files.
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
data_file = os.path.join(input_dir, "loan_data.csv")

#BU_Id Constant used in random seed methods
BU_ID = 4963

#Define the dataframe for the summarized table
df_summary = pd.DataFrame(
    columns = ["Model", "TP", "FP", "TN", "FN", "Accuracy","TPR", "TNR", "F1[0]", "F1[1]"]
)

def plot_credit_history(df_data):
    """
    This method plots the distribution of the credit_history feature when 
    the loan_approval is positive (Y:1)
    
    Args:
        df_data(Dataframe): It contains the loan data.
    """
    # Filter data where Loan_Status is "Y"
    filtered_df = df_data[df_data['Loan_Status'] == 1]
    # Count the frequency of Credit_History values
    credit_history_counts = filtered_df['Credit_History'].value_counts()
    # Plotting a pie chart
    bar_chart = plt.figure(figsize=(6, 6))
    plt.pie(credit_history_counts, labels=credit_history_counts.index, 
            autopct='%1.1f%%', startangle=90)
    plt.title('Credit History Distribution for Loan_Status="Y"')
    bar_chart.figure.savefig(os.path.join(input_dir,
                                         "df_data_credit_history_dist.png"))
    #plt.show()

def load_data():
    """
    Loads, processes and returns the df for loan information.
    Returns:
        df_data (Dataframe): Dataframe that contains loan information
    """

    df_data = pd.read_csv(data_file)
    print(df_data.describe())
    print(df_data.dtypes)
    #Drop Loan_ID field as it is not useful
    df_data = df_data.drop(["Loan_ID"], axis=1)
    #Identify missing data
    print("Missing data")  
    print(df_data.isnull().sum())
    label_encoder = LabelEncoder()
    #Pre-process the dataset to correctly fill missing data
    df_data['Gender'] = label_encoder.fit_transform(df_data['Gender'])
    df_data['Married'] = label_encoder.fit_transform(df_data['Married'])
    df_data['Education'] = label_encoder.fit_transform(df_data['Education'])
    #df_data['Self_Employed'] = df_data['Self_Employed'].map({'Y': 1, 'N': 0})
    df_data['Self_Employed'] = label_encoder.fit_transform(df_data['Self_Employed'])
    df_data['Property_Area'] = df_data['Property_Area'].map({'Rural': 0, 
                                                             'Semiurban': 1,
                                                             'Urban': "2"})
    df_data['Loan_Status'] = df_data['Loan_Status'].map({'Y': 1, 'N': 0})
    #Treat missing data
    df_data['Self_Employed'] = df_data['Self_Employed'].fillna(value="0") 
    # Replace missing values with the mode or a default value
    df_data['Credit_History'] = df_data['Credit_History'].fillna(df_data['Credit_History'].mode()[0])
    df_data['Gender'] = df_data['Gender'].fillna(value="1")   
    df_data['Dependents'] = df_data['Dependents'].fillna(value="0")  
    df_data.loc[(df_data['Dependents'] == "3+"), "Dependents"] = 3
    df_data["Loan_Amount_Term"] = df_data["Loan_Amount_Term"].fillna(
        df_data["Loan_Amount_Term"].median())
    #Check if there is any other missing data
    print("Total missing data after pre-processing")
    print(df_data.isnull().sum())
    # construct correlation matrix
    print("Correlation matrix")
    print(df_data.corr())
    plt.figure(figsize=(10,10))
    plt.title("Correlation matrix for Loan Data")
    sns_heat = sns.heatmap(df_data.corr(), annot=True,cmap='YlGnBu', fmt=".2f")
    sns_heat.set_xticklabels(sns_heat.get_xticklabels(), rotation=45, 
                             fontsize=7, rotation_mode='anchor', ha='right')    
    sns_heat.figure.savefig(os.path.join(input_dir,
                                         "df_data_heat_map_cormatrix.png"))
    
    #plt.show()
    # plot distribution of credit_history feature to select the correct values
    # for missing values
    plot_credit_history(df_data)

    return df_data



def compute_summary_conf(model_eval, y_test, y_pred):
    """
    Computes summary table for all the metrics: TP, FP, FP,...
    This is a variation of compute_summary but based on Confusion Matrix.
    This function also writes the summary to the df_summary.csv file for
    post-analysis.

    Args:
        model_eval (str): Model to be evaluated
        y_test(Dataframe): y test dataframe.
        y_pred(Dataframe): y prediction dataframe.

    """

    conf_matrix = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    # Get true positive rate (TPR)
    TPR = TP / (TP + FN)
    # Compute true negative rate (TNR) 
    TNR = TN / (TN + FP)
    # f1 Score
    f1_scores = f1_score(y_test, y_pred, average=None)
    f1_score_class_0 = f1_scores[0]
    f1_score_class_1 = f1_scores[1]
    row = {
        "Model": model_eval,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Accuracy": accuracy_score(y_test, y_pred),
        "TPR": TPR,
        "TNR": TNR,
        "F1[0]": f1_score_class_0,
        "F1[1]": f1_score_class_1
        }
    # Add the summary row to the df_summary DataFrame
    df_summary.loc[len(df_summary)] = row
    df_summary.to_csv("df_summary.csv", mode='a', header=not 
                      os.path.exists("df_summary.csv"), index=False)
    return df_summary  


def evaluate_models(models_list, X_test, y_test):
    """
    Given a list of modesl, this methods evaluates each model by predicting
    values on the X_test dataframe.

    Args:
        models_list(List): Model to be evaluated
        y_test(Dataframe): y test dataframe.
        y_pred(Dataframe): y prediction dataframe.

    """
    # Models to evaluate
    for model in models_list:
        model_name = type(model).__name__
        print(f"Model to evaluate:{model_name}")
        y_pred = model.predict(X_test)
        acc_model = accuracy_score(y_test, y_pred)
        print(f'Accuracy for {model}: , {acc_model}')
        conf_mat = confusion_matrix(y_test, y_pred)
        df_summary = compute_summary_conf(model_name, y_test, y_pred)
        print(f"Classification report for {model_name}")
        print(classification_report(y_test, y_pred))
        plt.figure(figsize=(8, 8))
        sns_heat = sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        sns_heat.figure.savefig(os.path.join(input_dir,
                                         model_name + "_confusion.png"))
        #plt.show()
    print(round(df_summary,3))