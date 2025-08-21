import numpy as np

class LinRegression:
    
    def __init__(self, X_dict, y, intercept=False):
        self.models = X_dict.keys()
        self.X_dict = X_dict
        self.y = y
        self.intercept = intercept
        if self.intercept:
            self.X_dict = {model: np.column_stack((np.ones(X.shape[0]), X)) for model, X in X_dict.items()}
        self.coefs = self.fit()
        self.predictions = self.predict()
        self.scores = self.score()

    def fit(self):
        coefs = {}
        for model in self.models:
            X = self.X_dict[model]
            # Fit the model using least squares
            coefs[model] = np.linalg.lstsq(X, self.y, rcond=None)[0]
        return coefs
    
    def predict(self):
        predictions = {}
        for model in self.models:
            X = self.X_dict[model]
            predictions[model] = X @ self.coefs[model]
        return predictions
    
    def score(self):
        scores = {}
        for model in self.models:
            metrics = {}
            metrics['mse'] = np.mean((self.y - self.predictions[model]) ** 2)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = np.mean(np.abs(self.y - self.predictions[model]))
            metrics['r2'] = 1 - (np.sum((self.y - self.predictions[model]) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2))
            metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (len(self.y) - 1) / (len(self.y) - self.X_dict[model].shape[1] - 1)
            scores[model] = metrics
        return scores

    def get_coefficients(self):
        return self.coefs

    def get_scores(self):
        return self.scores
