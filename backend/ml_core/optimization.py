"""
ML Core Optimization: Hyperparameter search and optimization
"""
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def grid_search(model, param_grid, X, y, scoring='neg_mean_squared_error', cv=3):
    gs = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
    gs.fit(X, y)
    return gs.best_params_, gs.best_score_

def random_search(model, param_dist, X, y, scoring='neg_mean_squared_error', cv=3, n_iter=10):
    rs = RandomizedSearchCV(model, param_dist, n_iter=n_iter, scoring=scoring, cv=cv)
    rs.fit(X, y)
    return rs.best_params_, rs.best_score_ 