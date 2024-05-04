"""
Maria Teresa Luna Plascencia
Class: CS 677
Date: 4/23/2024
Final Project: Loan Approval Predictor - Evaluation of various ML models
Description of Problem (just a 1-2 line summary!):
    Evaluate various classification ML models to identify the most suitable
    for predicting loan approvals.
"""

import helper
from rf_classifier import RFClassifier 
from knn_classiffier import KnnClassifier 
from svm_classifier import SVMClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


try:
        # Load and process loan file
        print(("Loading loan file into a pandas dataframe"))
        # Load the data from the loan file
        df_data = helper.load_data()
        print(df_data.head())

        #Feature matrix: Include all columns except the last one pertaining to the
        #target var.
        #include all columns except the target var
        X = df_data.iloc[:,:-1]
        #only include the target var
        y = df_data.iloc[:,-1]   
        X_train,X_test,y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                random_state=helper.BU_ID, 
                                                stratify=y) 
        print("X TRAIN")
        print(X_train.shape)
        print("X TEST")
        print(X_test.shape)
        #Scale data for certain models
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_sc = scaler.transform(X_train)
        X_test_sc = scaler.transform(X_test)

        #Instantiate RF Classifer for evaluation of Random Forest clf
        rf_clf = RFClassifier(X_train_sc, y_train, X_test_sc, y_test, 15, 5)
        best_rf = rf_clf.eval_RF(0)

        #Instantiate Balanced RF Classifer for evaluation of Balanced
        #Random Forest clf
        brf_clf = RFClassifier(X_train_sc, y_train, X_test_sc, y_test, 15, 5)
        best_brf = rf_clf.eval_RF(1)

        #Instantiate KNN Classifer for evaluation of KNN
        knn_clf = KnnClassifier(X_train_sc, y_train, X_test_sc, y_test, 
                                list(range(1, 21)))
        best_knn = knn_clf.eval_knn()

        #Instantiate logistic regression classifier
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train_sc, y_train)

        #Decision Tree
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(X_train_sc, y_train)

        #Try various SVM models and use the best one specifically 
        svm_clf = SVMClassifier(X_train_sc, y_train, X_test_sc, y_test)
        best_svm = svm_clf.try_SVM_models()

        # Define the VotingClassifier
        voting_clf = VotingClassifier(estimators=[
        ('rf', best_rf),
        ('brf', best_brf),
        ('knn', best_knn),
        ('svm', best_svm)
        ], voting='hard')

        # Train and evaluate the VotingClassifier
        voting_clf.fit(X_train_sc, y_train)

        models_evaluate = [best_rf, best_brf, best_knn, log_reg, best_svm, voting_clf, dt_clf]
        helper.evaluate_models(models_evaluate, X_test_sc, y_test)

except Exception as e:
    print(e)









