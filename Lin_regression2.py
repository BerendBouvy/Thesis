import numpy as np

class LinRegression:

    def __init__(self, X_dict_train, X_dict_test, y_train, y_test, intercept=False):
        self.models = X_dict_train.keys()
        self.X_dict_train = X_dict_train
        self.X_dict_test = X_dict_test
        self.y_train = y_train
        self.y_test = y_test
        self.intercept = intercept
        if self.intercept:
            self.X_dict_train = {model: np.column_stack((np.ones(X.shape[0]), X)) for model, X in X_dict_train.items()}
            self.X_dict_test = {model: np.column_stack((np.ones(X.shape[0]), X)) for model, X in X_dict_test.items()}
        self.coefs = self.fit()
        self.predictions = self.predict()
        self.scores = self.score()

    def fit(self):
        coefs = {}
        for model in self.models:
            X = self.X_dict_train[model]
            # Fit the model using least squares
            coefs[model] = np.linalg.lstsq(X, self.y_train, rcond=None)[0]
        return coefs
    
    def predict(self):
        predictions = {}
        for model in self.models:
            X = self.X_dict_test[model]
            predictions[model] = X @ self.coefs[model]
        return predictions
    
    def score(self):
        scores = {}
        for model in self.models:
            metrics = {}
            # metrics['mse'] = np.mean((self.y_test - self.predictions[model]) ** 2)
            metrics['rmse'] = np.sqrt(np.mean((self.y_test - self.predictions[model]) ** 2))
            # metrics['mae'] = np.mean(np.abs(self.y_test - self.predictions[model]))
            # metrics['r2'] = 1 - (np.sum((self.y_test - self.predictions[model]) ** 2) / np.sum((self.y_test - np.mean(self.y_test)) ** 2))
            # metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (len(self.y_test) - 1) / (len(self.y_test) - self.X_dict_test[model].shape[1] - 1)
            # metrics['bic'] = len(self.y_test) * np.log(metrics['mse']) + (self.X_dict_test[model].shape[1] + 1) * np.log(len(self.y_test))
            # metrics['aic'] = len(self.y_test) * np.log(metrics['mse']) + 2 * (self.X_dict_test[model].shape[1] + 1)
            metrics['predicted R^2'] = 1 - (np.sum((self.y_test - self.predictions[model]) ** 2) / np.sum((self.y_test - np.mean(self.y_test)) ** 2))
            scores[model] = metrics
        return scores

    def get_coefficients(self):
        return self.coefs

    def get_scores(self):
        return self.scores
