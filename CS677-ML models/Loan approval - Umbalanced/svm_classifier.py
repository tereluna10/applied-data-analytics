"""
Maria Teresa Luna Plascencia
Class: CS 677
Date: 4/23/2024
Final Project: Loan Approval Predictor - Evaluation of various ML models
Description of Problem (just a 1-2 line summary!):
    This class defines the abstraction of a Random Forest Classifier by
    encapsulating all the model parameters associated to this model.
"""

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVMClassifier:
    def __init__ (self, X_train, y_train, X_test, y_test):
        """
        Constructor of the Random Forest Classifier/
        Args:
            self:
            X_train:
            y_train:
            X_test:
            y_test:
            d:
            N:
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # Define parameters for SVM models
        self.c = 10  # Regularization parameter
        self.SVC_models = [
            LinearSVC(C=self.c, loss="hinge"),
            SVC(kernel='rbf', C=self.c), #
            SVC(kernel='poly', C=self.c, degree = 3),
        ]

    def try_SVM_models(self):
        """
        It trains the data using various SMV models define in the SVC_model
        attribute 
        var models.

        Args:
            self
        Returns:
            An instance of the best SVM classifier found.
        Returns:
            best_svm(LinearSVC): Returns the best SVM classifier object
        """
        best_accuracy = 0
        best_svm = None
        for model in self.SVC_models:
            model_name = type(model).__name__
            # For non-linear kernels, include kernel information
            if isinstance(model, SVC) and model.kernel != 'linear':
                model_name += f" ({model.kernel} kernel)"
            print("Evaluating ", model_name)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            acc_model = accuracy_score(self.y_test,y_pred)
            #print("Accuracy: ", acc_model)
            if acc_model > best_accuracy:
                best_accuracy = acc_model
                best_svm = model
        return best_svm
    