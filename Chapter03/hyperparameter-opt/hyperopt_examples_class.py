from hyperopt import hp, tpe, STATUS_OK, Trials, fmin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import pprint


class LogisticRegressionOptimizer:
    def __init__(self, n_folds, n_samples, n_features, n_informative, n_redundant, train_samples):
        self.n_folds = n_folds
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.train_samples = train_samples

    @staticmethod
    def objective(params, n_folds, X, y):
        clf = LogisticRegression(**params, random_state=42)
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='f1_macro')
        max_score = max(scores)
        loss = 1 - max_score
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    def optimize(self):
        X, y = datasets.make_classification(n_samples=self.n_samples, n_features=self.n_features,
                                              n_informative=self.n_informative, n_redundant=self.n_redundant)
        X_train = X[:self.train_samples]
        y_train = y[:self.train_samples]
        space = {
            'warm_start': hp.choice('warm_start', [True, False]),
            'fit_intercept': hp.choice('fit_intercept', [True, False]),
            'tol': hp.uniform('tol', 0.00001, 0.0001),
            'C': hp.uniform('C', 0.05, 2.5),
            'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear']),
            'max_iter': hp.choice('max_iter', range(10, 500))
        }
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=16,
            trials=trials,
            args=(self.n_folds, X_train, y_train)
        )
        return best


if __name__ == '__main__':
    optimizer = LogisticRegressionOptimizer(
        n_folds=5,
        n_samples=100000,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        train_samples=100
    )
    best_params = optimizer.optimize()
    pprint.pprint(best_params)
