# Try to import sklearn components with fallback
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback implementations if sklearn is not available
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            return self
        def transform(self, X):
            return (X - self.mean_) / self.scale_
        def fit_transform(self, X):
            return self.fit(X).transform(X)
    
    class MinMaxScaler:
        def __init__(self):
            self.min_ = None
            self.scale_ = None
        def fit(self, X):
            self.min_ = np.min(X, axis=0)
            self.scale_ = np.max(X, axis=0) - self.min_
            return self
        def transform(self, X):
            return (X - self.min_) / self.scale_
        def fit_transform(self, X):
            return self.fit(X).transform(X)
    
    class IsolationForest:
        def __init__(self, **kwargs):
            pass
        def fit_predict(self, X):
            return np.zeros(len(X))
    
    SKLEARN_AVAILABLE = False 