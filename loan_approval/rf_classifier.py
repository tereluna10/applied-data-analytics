"""
Maria Teresa Luna Plascencia
Class: CS 677
Date: 4/23/2024
Final Project: Loan Approval Predictor - Evaluation of various ML models
Description of Problem (just a 1-2 line summary!):
    This class defines the abstraction of a Random Forest Classifier 
    and Balanced Random Forest Classifier by
    encapsulating all the model parameters associated to this model.
"""

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RFClassifier:
    def __init__ (self, X_train, y_train, X_test, y_test, num_subtrees, max_depth):
        """
        Constructor of the Random Forest Classifier/
        Args:
            self:
            X_train (Dataframe): X train set.  It can be scaled, depending on 
            how it is sent from outside callers.
            y_train (Dataframe): y train set.
            X_test (Dataframe): X test set.  It can be scaled, depending on 
            how it is sent from outside callers.
            y_test (Dataframe):  y test set.
            d (int): Max Depth of the tree to try
            N (int): Max Number of subrees to try
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_subtrees = num_subtrees
        self.max_depth = max_depth


    def plot_accu(self, df_accu, rf_type):
        """
        It plots the accuracy of a list of numbers using a line chart

        Args:
            df_accu (Dataframe): It contains the accuracy list.
            rf_type(int): 1 - Balanced Random Forest, 0 Random Forest
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_accu, markers=True)

        # Add labels and title
        plt.xlabel('N - number of subtrees')
        plt.ylabel('Accuracy')
        plt.title('Random Forest - Acurracy Plot')

        # Show the plot
        plt.grid(True)
        plt.legend(title='d-max depth')
        if rf_type == 1:
            plt.savefig("RandomForest-BestNd.png")
        else:
            plt.savefig("BalancedRandomForest-BestNd.png")
        #plt.show()

    
    def eval_RF(self, rf_type):
        """
        Methods that applies Random Forest classifier to the dataframe, and
        finds the best hyperparameters for both number of subrees and max_depth.
        Args:
            rf_type(int): 1 - Balanced Random Forest, 0 Random Forest
        Returns:
            best_rf(BalancedRandomForestClassifier | RandomForestClassifier):
            Returns a instance of any of these classifier, the best found.
        """
        best_accuracy = 0
        best_N = 0
        best_d = 0
        #Dataframe to store the various accuracy numbers after executing
        #the models
        df_acc = pd.DataFrame(
            columns = ["d" + str(i) for i in range (1,self.max_depth)],
            index = ["N" + str(i) for i in range (1,self.num_subtrees)],
            dtype = float
        )
        for N in range (1, self.num_subtrees):
            for d in range (1, self.max_depth):
                if rf_type == 1:
                    clf = BalancedRandomForestClassifier(n_estimators = N, criterion = "entropy",
                                                random_state = 42,
                                                max_depth = d, sampling_strategy = "all",
                                                replacement = True,
                                                bootstrap = False)
                else:
                    clf = RandomForestClassifier(n_estimators = N, criterion = "entropy",
                                                random_state = 42,
                                                max_depth = d)
                clf.fit(self.X_train, self.y_train)
                y_pred = clf.predict(self.X_test)
                acc_model = accuracy_score(self.y_test,y_pred)
                #print("Accuracy : ", acc_model)
                if acc_model > best_accuracy:
                    best_accuracy = acc_model
                    best_N = N
                    best_d = d
                df_acc.loc["N"+str(N), "d"+str(d)] = acc_model
        print(df_acc)
        #print("METRICS***************")
        #print(metrics.classification_report(self.y_pred, self.y_test))
        # Plot df_accu
        self.plot_accu(df_acc, rf_type)
        print(f"Best combination: N={best_N}, d={best_d}, Acc: {best_accuracy}")
        if rf_type == 1:
            best_rf = BalancedRandomForestClassifier(n_estimators = best_N, criterion = "entropy",
                                                random_state = 42,
                                                max_depth = best_d, sampling_strategy = "all",
                                                replacement = True,
                                                bootstrap = False)  
        else:
            best_rf = RandomForestClassifier(n_estimators = best_N, criterion = "entropy",
                                            random_state = 42,
                                            max_depth = best_d)        
        best_rf.fit(self.X_train, self.y_train)
        return best_rf

