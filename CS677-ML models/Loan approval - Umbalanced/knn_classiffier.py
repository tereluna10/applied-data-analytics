"""
Maria Teresa Luna Plascencia
Class: CS 677
Date: 4/23/2024
Final Project: Loan Approval Predictor - Evaluation of various ML models
Description of Problem (just a 1-2 line summary!):
    This class defines the abstraction of a Random Forest Classifier by
    encapsulating all the model parameters associated to this model.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class KnnClassifier:
    def __init__ (self, X_train, y_train, X_test, y_test, k_neighbors):
        """
        Constructor of the KNN Classifier/
        Args:
            self:
            X_train:
            y_train:
            X_test:
            y_test:
            k_neighbors (list): List of the k values to try
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.k_neighbors = k_neighbors

    def plot_accu(self, accuracy_list):
        """
        
        """
        #Plot all the accuracies
        #Q3.2 Plotting a graph showing the different accuracies
        print("Q3.2 Plotting a graph showing the different accuracies")
        ax = plt.axes()
        ax.set_xticks(self.k_neighbors)
        plt.plot(self.k_neighbors, accuracy_list, "r-")
        plt.xlabel("k")
        plt.ylabel("Accuracy of Predictions")
        plt.title("Plot of Accuracy vs k value for kNN")
        #plt.show()
    
    def eval_knn(self):
        """
        Methods that applies Random Forest classifier to df_fet 
        Args:
            self
        Returns:
            best_knn(KNeighborsClassifier): The best classifier with the best
            k param. found.
        """
        best_accuracy = 0
        best_k = 0
        accuracy = []
        for k in self.k_neighbors:
            classifier = KNeighborsClassifier(n_neighbors=k,p=2,metric='euclidean') 
            classifier.fit(self.X_train,self.y_train)
            y_pred =  classifier.predict(self.X_test)
            accuracy.append(accuracy_score(self.y_test,y_pred))
            #print(f"Accuracy for k: {k}, {accuracy_score(self.y_test,y_pred)}")
            acc_model = accuracy_score(self.y_test,y_pred)
            #print("Accuracy : ", acc_model)
            if acc_model > best_accuracy:
                best_accuracy = acc_model
                best_k = k
        self.plot_accu(accuracy)
        best_knn = KNeighborsClassifier(n_neighbors=best_k,p=2,metric='euclidean') 
        best_knn.fit(self.X_train,self.y_train)
        return best_knn

